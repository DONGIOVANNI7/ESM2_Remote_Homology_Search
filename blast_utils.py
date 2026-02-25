from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BlastHit:
    """
    Represents one BLAST hit (one alignment row) for a query sequence.
    We keep only the fields we need for ranking and reporting.
    """
    subject: str # subject protein ID (database sequence)
    pident: float # percent identity (0..100)
    bitscore: float # BLAST bit score (higher = better match)
    evalue: float # E-value (lower = more significant)


def norm_id(x: str) -> str:
    """
    Normalize BLAST identifiers so they match the IDs we use elsewhere.

    BLAST (outfmt 6) may output IDs like:
      sp|P12345|PROT_HUMAN
      tr|A0A0B4...|SOME_NAME

    Our pipeline typically uses the accession (the middle field: P12345 / A0A0B4...),
    so this function extracts that when '|' is present.
    If the string has no pipes, return it unchanged.
    """
    if "|" in x:
        parts = x.split("|")
        # SwissProt / TrEMBL formats: sp|ACC|NAME or tr|ACC|NAME
        if len(parts) >= 2 and parts[1]:
            x = parts[1]
    return x


def parse_blast_outfmt6(
    path: str,
    evalue_max: float = 1e-3
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
    """
    Parse a BLAST tabular output file (outfmt 6) and return two structures:

    1) top:
       Dict[query_id -> List[subject_id]]
       For each query, a list of subject IDs sorted by decreasing bitscore.

    2) pident_map:
       Dict[(query_id, subject_id) -> pident]
       A lookup table to quickly retrieve percent identity for a (query, subject) pair.

    Notes:
    - This parser expects at least 12 columns, which matches the default outfmt 6
      columns: qacc sacc pident length mismatch gapopen qstart qend sstart send evalue bitscore
    - We apply an E-value cutoff (evalue_max). Hits above the threshold are ignored
      and will not appear in either output structure.
    """
    # Collect hits grouped by query
    by_q: Dict[str, List[BlastHit]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 12: # Need at least the standard 12 fields so indexes 10 and 11 exist
                continue
            q, s = norm_id(parts[0]), norm_id(parts[1]) # Normalize both query and subject IDs (e.g., sp|ACC|NAME -> ACC)
            # Parse numeric columns; skip malformed lines
            try:
                pident = float(parts[2])
                evalue = float(parts[10])
                bitscore = float(parts[11])
            except ValueError:
                continue
            if evalue > evalue_max:
                continue
            hit = BlastHit(s, pident, bitscore, evalue)
            by_q.setdefault(q, []).append(hit)

    top: Dict[str, List[str]] = {}
    pident_map: Dict[Tuple[str, str], float] = {}
    for q, hits in by_q.items():
        # Rank by bitscore: highest bitscore first
        hits.sort(key=lambda h: h.bitscore, reverse=True)
        # Subject ranking list for this query
        top[q] = [h.subject for h in hits]
        # Quick lookup for pident per (query, subject)
        # If BLAST produced duplicate rows for the same (q,s), this will keep the last one
        # in the sorted list order (which should be among the best if duplicates exist).
        for h in hits:
            pident_map[(q, h.subject)] = h.pident
    return top, pident_map
