from dataclasses import dataclass
from typing import Iterator, Optional, List


@dataclass
class FastaRecord:
    """
    Simple container for one FASTA entry.

    FASTA format:
      >header line (metadata)
      SEQUENCE....
      SEQUENCE....
    """
    header: str # full header line, including the leading ">"
    seq: str # sequence string (concatenated, no whitespace)

    @property
    def id(self) -> str:
        """
        Extract a stable identifier from the FASTA header.

        Common UniProt-style headers look like:
          >sp|P12345|PROT_HUMAN Some description...
          >tr|A0A0B4J2D5|SOME_NAME ...
        In that case we return the accession (the 2nd pipe-separated field),
        e.g. "P12345" or "A0A0B4J2D5".

        If the header doesn't contain pipes, we fall back to the first
        whitespace-separated token after ">".
        """
        h = self.header.strip()
        if h.startswith(">"):
            h = h[1:] # remove FASTA '>' prefix
        parts = h.split("|")
        # UniProt headers: parts[1] is the accession
        if len(parts) >= 2 and parts[1]:
            return parts[1].strip()
        return h.split()[0].strip() # Generic FASTA header: use first token as id

    @property
    def description(self) -> str:
        """
        Extract a short "description/name" from the header.

        For UniProt-like headers:
          >sp|P12345|PROT_HUMAN ...
        parts[2] is often the entry name (e.g. "PROT_HUMAN").

        If not available, return empty string.
        """
        h = self.header.strip()
        if h.startswith(">"):
            h = h[1:]
        parts = h.split("|")
        if len(parts) >= 3 and parts[2]:
            return parts[2].strip()
        return ""


def iter_fasta(path: str) -> Iterator[FastaRecord]:
    """
    Stream a FASTA file record-by-record (generator).

    - Reads the file line by line (memory-friendly).
    - Each time we encounter a new header line ('>'), we yield the previous record.
    - Sequence lines are accumulated and concatenated into a single string.
    """
    header: Optional[str] = None # current record header (None means "no record yet")
    seq_chunks: List[str] = [] # list of sequence lines for the current record
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, seq="".join(seq_chunks))
                header = line
                seq_chunks = [] # reset sequence accumulator for the new record
            else:
                seq_chunks.append(line)
        if header is not None:
            yield FastaRecord(header=header, seq="".join(seq_chunks))
