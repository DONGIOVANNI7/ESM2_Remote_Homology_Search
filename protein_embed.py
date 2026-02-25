#!/usr/bin/env python3
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
from protein_fasta import iter_fasta
from esm_embedder import ESM2Embedder, EmbedConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", default="esm2_t6_8M_UR50D")
    ap.add_argument("--max_aa", type=int, default=1022)
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )
    ap.add_argument("--max_tokens_per_batch", type=int, default=8000)
    args = ap.parse_args()

    out = Path(args.output)
    # Store IDs and metadata alongside vectors to keep row-to-protein mapping
    ids_path = Path(str(out) + ".ids.txt") # one protein ID per line (same order as vectors)
    meta_path = Path(str(out) + ".meta.tsv") # TSV with id, length, description

    cfg = EmbedConfig(
        model_name=args.model,
        max_aa=args.max_aa,
        repr_layer=6,   # repr_layer selects which transformer layer representation to extract
        device=args.device,
        max_tokens_per_batch=args.max_tokens_per_batch
    )
    emb = ESM2Embedder(cfg)
    # We do this to know the number of sequences (for tqdm total) and to write
    # a mapping file (ids / meta) that aligns with the vector rows
    ids = []
    meta = []
    for r in iter_fasta(args.input):
        # Truncate sequences that exceed model's maximum supported length
        seq = r.seq[:args.max_aa] if len(r.seq) > args.max_aa else r.seq
        ids.append(r.id)
        # Store basic metadata for analysis/debugging
        meta.append((r.id, len(seq), r.description))
    # embed_records(...) yields (protein_id, vector) pairs. We build a list and stack into
    # a single 2D array: shape = (num_proteins, embedding_dim)
    vecs = []
    for pid, v in tqdm(emb.embed_records(iter_fasta(args.input)), total=len(ids)):
        vecs.append(v)
    # Stack list of vectors into one dense array and force float32 for compact storage
    X = np.stack(vecs, axis=0).astype(np.float32)

    with open(out, "wb") as f:
        np.save(f, X) # Save vectors in NumPy binary format
    # Save ID list (one per line). This allows you to map row i -> protein ID
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    # Save metadata TSV (id, truncated length, description). Replace tabs in description
    # so the TSV stays well-formed.
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("id\tlength\tdescription\n")
        for pid, L, desc in meta:
            safe_desc = (desc or "").replace("\t", " ")
            f.write(f"{pid}\t{L}\t{safe_desc}\n")

    print(f"[OK] vectors: {out} shape={X.shape}")
    print(f"[OK] ids:     {ids_path}")
    print(f"[OK] meta:    {meta_path}")


if __name__ == "__main__":
    main()
