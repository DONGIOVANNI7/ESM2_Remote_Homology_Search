from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple
import numpy as np
import torch
from protein_fasta import FastaRecord


@dataclass
class EmbedConfig:
    """
    Configuration for ESM2 embedding.

    - model_name: which pretrained ESM2 model to load (from fair-esm).
    - max_aa: maximum amino-acid length allowed by the model (longer sequences are truncated).
    - repr_layer: which transformer layer's representations to use as embeddings.
    - device: "auto" picks CUDA if available else CPU; can also force "cpu" or "cuda".
    - max_tokens_per_batch: controls dynamic batching based on total tokens, not number of sequences.
      This is important because transformer cost scales with sequence length.
    """
    model_name: str = "esm2_t6_8M_UR50D"
    max_aa: int = 1022
    repr_layer: int = 6
    device: str = "auto"
    max_tokens_per_batch: int = 8000


class ESM2Embedder:
    """
    Wrapper around a pretrained ESM2 model that produces one vector per protein sequence.

    Strategy used here:
    - For each sequence, get token-level representations from the selected transformer layer.
    - Compute a single fixed-size embedding by averaging token vectors across the amino acids
      (mean pooling), excluding special tokens (<cls>, <eos>) and padding.
    """
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        # fair-esm provides the ESM2 models and tokenizer / alphabet utilities
        try:
            import esm
        except Exception as e:
            raise RuntimeError("Install fair-esm: pip install fair-esm") from e
        # Look up the loader function for the chosen model name
        loader = getattr(esm.pretrained, cfg.model_name, None)
        if loader is None:
            raise ValueError(f"Unknown ESM model: {cfg.model_name}")
        # Load pretrained model and its alphabet (tokenization scheme)
        model, alphabet = loader()
        self.model = model
        self.alphabet = alphabet
        # batch_converter converts [(label, seq), ...] into (labels, strs, tokens)
        # where tokens is a padded tensor of token IDs.
        self.batch_converter = alphabet.get_batch_converter()

        if cfg.device == "auto":
            self.device = torch.device(
                              "cuda" if torch.cuda.is_available() else "cpu"
                          )
        else:
            self.device = torch.device(cfg.device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _truncate(self, seq: str) -> str: # Clean and truncate a protein sequence to fit model constraints.
        seq = seq.strip()
        return seq[: self.cfg.max_aa] if len(seq) > self.cfg.max_aa else seq

    def embed_records(
        self,
        records: Iterable[FastaRecord]
    ) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Embed a stream of FASTA records and yield (protein_id, embedding_vector).

        This function uses "dynamic batching":
        - It accumulates sequences until the total number of tokens in the batch would exceed
          max_tokens_per_batch.
        - Then it runs one forward pass and yields vectors for that batch.
        This keeps GPU/CPU memory usage bounded and improves throughput.
        """
        buf: List[Tuple[str, str]] = [] # Buffer of (protein_id, sequence_string) items to embed as a batch
        # Approximate token count accumulated in the buffer
        # (tokens ~= amino acids + 2 special tokens)
        buf_tokens = 0 

        def flush():
            """
            Run the model on the current buffered batch and yield embeddings.
            Resets the buffer afterwards.
            """
            nonlocal buf, buf_tokens
            if not buf:
                return
            # Convert raw sequences to padded token IDs understood by the model
            labels, strs, tokens = self.batch_converter(buf)
            tokens = tokens.to(self.device)
            # Forward pass without gradient tracking (faster + lower memory)
            with torch.no_grad():
                out = self.model(
                    tokens,
                    repr_layers=[self.cfg.repr_layer],
                    return_contacts=False
                )
            # Extract hidden representations for the requested layer
            # reps shape: [B, T, d]
            #   B = batch size
            #   T = token length (includes special tokens and padding)
            #   d = embedding dimension
            reps = out["representations"][self.cfg.repr_layer]  # [B,T,d]
            # Convert each sequence in the batch into a single vector
            for i, (pid, seq) in enumerate(buf):
                L = len(seq) # number of amino acids (no special tokens)
                # Token layout (typical ESM):
                # index 0: <cls>
                # next L positions: amino acids
                # then <eos>, then padding
                #
                # We average only the amino-acid token representations:
                # reps[i, 1:1+L, :]
                vec = reps[i, 1:1+L, :].mean(dim=0)  # exclude <cls>,<eos>,pads
                yield pid, vec.detach().cpu().numpy().astype(np.float32)
            buf = []
            buf_tokens = 0

        for i, r in enumerate(records):
            pid = r.id
            seq = self._truncate(r.seq)
            # Approximate tokens for batching:
            # amino acids + 2 (for <cls> and <eos>)
            t = len(seq) + 2
            # If adding this sequence would exceed our batch token budget,
            # flush current batch first.
            if buf and (buf_tokens + t) > self.cfg.max_tokens_per_batch:
                yield from flush()
            # Add to buffer
            buf.append((pid, seq))
            buf_tokens += t
        # Flush whatever is left at the end
        yield from flush()
