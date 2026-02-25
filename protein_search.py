#!/usr/bin/env python3
import argparse
import time
import os
import subprocess
import hashlib
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from protein_fasta import iter_fasta
from esm_embedder import ESM2Embedder, EmbedConfig
from blast_utils import parse_blast_outfmt6
from ann import EuclideanLSH, LSHParams
from ann import HypercubeIndex, HypercubeParams
from ann import IVFFlat, IVFParams
from ann import IVFPQ, IVFPQParams
from ann import NeuralLSH, NeuralParams
from ann import ANNIndex, ANN_METHODS
from ann import ANNStatistics


def load_vectors(p: str) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """
    Load the database embedding matrix and its metadata.

    Expected files:
      p -> a .npy file (written by np.save)
      p + ".ids.txt" -> one protein ID per line (row-aligned with X)
      p + ".meta.tsv" (opt.) -> tab-separated: id, length, description

    Returns:
      X: [N, d] float32 matrix of protein vectors
      ids: list of N protein IDs aligned with X rows
      desc: dict mapping protein ID -> description (if meta exists)
    """
    fp = Path(p)
    with open(fp, "rb") as f:
        X = np.load(f).astype(np.float32)
    # Row-aligned IDs (same order as vectors)
    ids = Path(str(fp) + ".ids.txt").read_text(encoding="utf-8").splitlines()
    desc: Dict[str, str] = {}
    meta = Path(str(fp) + ".meta.tsv")
    if meta.exists():
        with open(meta, "r", encoding="utf-8") as f:
            f.readline() # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    desc[parts[0]] = parts[2]
    return X, ids, desc


def run_blast(db_fasta: str, query_fasta: str, out_tsv: str) -> Tuple[float, float]:
    """
    Build a local BLAST database and run blastp for the query FASTA.

    Output:
      out_tsv: BLAST tabular output (outfmt 6) written to file

    Returns:
      (makeblastdb_seconds, blastp_seconds)
    """
    db_prefix = str(Path(out_tsv).with_suffix("")) + "_db"
    print("Running makeblastdb")
    t0 = time.perf_counter()
    subprocess.run(
        ["makeblastdb", "-in", db_fasta, "-dbtype", "prot", "-out", db_prefix],
        check=True
    )
    make_s = time.perf_counter() - t0
    print("Running blastp")
    t1 = time.perf_counter()
    subprocess.run([
        "blastp",
        "-db", db_prefix,
        "-query", query_fasta,
        "-outfmt", "6",
        "-out", out_tsv,
        "-evalue", str(10),
        "-max_target_seqs", "50000",
        # "-max_hsps", "1",
        "-num_threads", str(os.cpu_count())
    ],
        check=True
    )
    blast_s = time.perf_counter() - t1
    return make_s, blast_s

def cache_key(name: str, params: dict, shape: Tuple[int, int]) -> str:
    """
    Create a stable cache key for an index based on:
      - method name
      - parameters
      - dataset shape (N, d)

    We hash these into a short string so cache filenames stay manageable.
    """
    h = hashlib.sha1(repr({"n": name, "p": params, "s": shape})
                     .encode("utf-8")).hexdigest()[:16]
    return f"{name}_{h}"


def cache_load_or_build(
    cache_dir: Path,
    key: str,
    indexer: ANNIndex,
    base: np.ndarray
) -> ANNIndex:
    """
    Load a previously-built ANN index from pickle cache if present.
    Otherwise build it from scratch, save it, and return it.

    This avoids rebuilding expensive indices (IVF, PQ, NeuralLSH) on every run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{key}.pkl"
    if fp.exists():
        print("Loading index from cache")
        with open(fp, "rb") as f:
            return pickle.load(f)

    print("Building index")
    indexer.build(base)
    with open(fp, "wb") as f:
        pickle.dump(indexer, f)
    return indexer


def fmt_pident(x: Optional[float]) -> str:
    """
    Format BLAST percent identity for display.
    If missing (no entry in blast_pident), prints '--'.
    """
    return "--" if x is None else f"{x:.2f}%"


def arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Input vectors + queries + report output
    ap.add_argument("-d", "--data", required=True)
    ap.add_argument("-q", "--queries", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "-method",
        "--method",
        required=True,
        choices=["all", *ANN_METHODS.keys()]
    )
    # Evaluation/printing controls
    ap.add_argument("--N_eval", type=int, default=50)
    ap.add_argument("--N_print", type=int, default=10)
    # BLAST control (ground truth + identity lookup)
    ap.add_argument("--blast_tsv", default=None)
    ap.add_argument("--run_blast", action="store_true")
    ap.add_argument("--db_fasta", default=None)
    # ESM embedding config for queries
    ap.add_argument("--model", default="esm2_t6_8M_UR50D")
    ap.add_argument("--max_aa", type=int, default=1022)
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )
    # Index building controls
    ap.add_argument("--train_size", type=int, default=50000)
    ap.add_argument("--cache_dir", default=None)
    # LSH parameters
    ap.add_argument("--lsh_k", type=int, default=10)
    ap.add_argument("--lsh_L", type=int, default=20)
    ap.add_argument("--lsh_w", type=float, default=4.0)
    # Hypercube parameters
    ap.add_argument("--cube_k", type=int, default=14)
    ap.add_argument("--cube_M", type=int, default=2000)
    ap.add_argument("--cube_w", type=float, default=4.0)
    ap.add_argument("--cube_probes", type=int, default=10)
    # IVF parameters
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--nprobe", type=int, default=10)
    # IVFPQ parameters
    ap.add_argument("--pq_m", type=int, default=8)
    ap.add_argument("--pq_refine", action="store_true")
    # Neural LSH parameters
    ap.add_argument("--neural_m", type=int, default=2048)
    ap.add_argument("--neural_T", type=int, default=5)
    ap.add_argument("--neural_epochs", type=int, default=5)

    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = arguments()

    X, base_ids, base_desc = load_vectors(args.data)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path(str(Path(args.data)) + ".cache")
    # Decide cache directory location
    blast_make_s: Optional[float] = None
    blast_search_s: Optional[float] = None
    if args.run_blast:
        if not args.db_fasta or not args.blast_tsv:
            raise ValueError("--run_blast needs --db_fasta and --blast_tsv")
        blast_make_s, blast_search_s = run_blast(args.db_fasta, args.queries, args.blast_tsv)
    # blast_top: query_id -> list of subject IDs ordered by bitscore (later sliced to N_eval)
    # blast_pident: (query_id, subject_id) -> percent identity
    blast_top: Dict[str, List[str]] = {}
    blast_pident: Dict[Tuple[str, str], float] = {}
    if args.blast_tsv:
        blast_top, blast_pident = parse_blast_outfmt6(args.blast_tsv, 1e-3)

    embedder = ESM2Embedder( 
        EmbedConfig(
            model_name=args.model,
            max_aa=args.max_aa,
            repr_layer=6,
            device=args.device,
            max_tokens_per_batch=4000
        )
    )

    qrecs = list(iter_fasta(args.queries)) # Load queries into memory because we embed them
    q_ids = [r.id for r in qrecs]
    q_vecs = [v for _, v in embedder.embed_records(qrecs)] # Embed each query, embed_records yields (id, vec) pairs
    Q = np.stack(q_vecs, axis=0).astype(np.float32)
    # BLAST timing (only available if we actually ran BLAST now)
    blast_time_per_query: Optional[float] = None
    blast_qps: Optional[float] = None
    if blast_search_s is not None and blast_search_s > 0 and len(q_ids) > 0:
        blast_time_per_query = blast_search_s / float(len(q_ids))
        blast_qps = float(len(q_ids)) / blast_search_s
    if args.method == "all":
        methods = ANN_METHODS.keys()
    else:
        methods = [args.method]

    idx_objs = {}
    for m in methods:
        if m == "lsh":
            print("LSH: ", end="")
            p = LSHParams(
                k=args.lsh_k,
                L=args.lsh_L,
                w=args.lsh_w,
                seed=args.seed
            )
            key = cache_key("lsh", asdict(p), X.shape)
            idx_objs[m] = cache_load_or_build(
                cache_dir,
                key,
                EuclideanLSH(p),
                X
            )
        elif m == "hypercube":
            print("Hypercube: ", end="")
            p = HypercubeParams(
                k=args.cube_k,
                M=args.cube_M,
                w=args.cube_w,
                probes=args.cube_probes,
                seed=args.seed
            )
            key = cache_key("cube", asdict(p), X.shape)
            idx_objs[m] = cache_load_or_build(
                cache_dir,
                key,
                HypercubeIndex(p),
                X
            )
        elif m == "ivfflat":
            print("IVFFlat: ", end="")
            p = IVFParams(
                nlist=args.nlist,
                nprobe=args.nprobe,
                train_size=args.train_size,
                seed=args.seed
            )
            key = cache_key("ivfflat", asdict(p), X.shape)
            idx_objs[m] = cache_load_or_build(
                cache_dir,
                key,
                IVFFlat(p),
                X
            )
        elif m == "ivfpq":
            print("IVFPQ: ", end="")
            p = IVFPQParams(
                nlist=args.nlist,
                nprobe=args.nprobe,
                m=args.pq_m,
                train_size=args.train_size,
                seed=args.seed,
                refine=args.pq_refine
            )
            key = cache_key("ivfpq", asdict(p), X.shape)
            idx_objs[m] = cache_load_or_build(
                cache_dir,
                key,
                IVFPQ(p),
                X
            )
        elif m == "neural":
            print("Neural LSH: ", end="")
            p = NeuralParams(
                m=args.neural_m,
                T=args.neural_T,
                epochs=args.neural_epochs,
                train_size=args.train_size,
                seed=args.seed
            )
            key = cache_key("neural", asdict(p), X.shape)
            idx_objs[m] = cache_load_or_build(
                cache_dir,
                key,
                NeuralLSH(p, device=args.device),
                X
            )

    with open(args.output, "w", encoding="utf-8") as f:
        stats: Dict[ANNStatistics] = {}
        blast_ref_sum = 0.0
        for qi, qid in enumerate(q_ids):
            q = Q[qi]
            f.write(f"Query Protein: {qid}\n")
            f.write(f"N = {args.N_eval} (Top-N used for Recall@N evaluation)\n\n")
            f.write("[1] Summary comparison\n")
            f.write("-"*70 + "\n")
            f.write("Method        | Time/query (s) | QPS     | Recall@N vs BLAST Top-N\n")
            f.write("-"*70 + "\n")

            blast_results = blast_top.get(qid, [])[:args.N_eval]
            blast_set = set(blast_results)
            blast_ref_sum += (1.0 if blast_results else 0.0)
            store = {}
            for m in methods:
                idx = idx_objs[m]
                t0 = time.perf_counter()
                nn_idx, nn_dist = idx.query(X, q, args.N_eval)
                dt = time.perf_counter() - t0
                qps = (1.0/dt) if dt > 0 else 0.0

                recall = None
                if blast_results:
                    ann_ids = [base_ids[int(i)] for i in nn_idx[:args.N_eval]]
                    recall = len(set(ann_ids).intersection(blast_set)) / float(len(blast_results))
                else:
                    recall = 0.0

                m_stats = stats.get(m, ANNStatistics())
                m_stats.qps += qps
                m_stats.recall += recall
                stats[m] = m_stats

                store[m] = (nn_idx, nn_dist, recall)
                label = ANN_METHODS[m]
                f.write(f"{label:13s} | {dt:14.4f} | {qps:7.2f} | {f'{recall:.2f}'}\n")
            # Add BLAST baseline row (reference)
            if args.blast_tsv:
                tpq_str = f"{blast_time_per_query:14.4f}" if blast_time_per_query is not None else f"{'--':>14}"
                qps_str = f"{blast_qps:7.2f}" if blast_qps is not None else f"{'--':>7}"
                rec_ref = 1.0 if blast_results else 0.0
                f.write(f"{'BLAST (Ref)':13s} | {tpq_str} | {qps_str} | {rec_ref:.2f}\n")
            f.write("-"*70 + "\n\n")
            f.write(f"[2] Top-N neighbors per method (printed N = {args.N_print})\n\n")

            for m in methods:
                nn_idx, nn_dist, recall = store[m]
                label = ANN_METHODS[m]
                f.write(f"Method: {label}\n")
                f.write("Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment\n")
                f.write("-"*90 + "\n")
                for r in range(min(args.N_print, nn_idx.size)):
                    bi = int(nn_idx[r])
                    # Convert neighbor index -> protein id (with bounds check)
                    nid = base_ids[bi] if 0 <= bi < len(base_ids) else str(bi)
                    dist = float(nn_dist[r])
                    # Lookup BLAST percent identity for (query, neighbor)
                    pid = blast_pident.get((qid, nid))
                    # Whether this neighbor is in BLAST Top-N for this query
                    in_blast = ("Yes" if nid in blast_set else "No") if blast_top else "--"
                    comment = base_desc.get(nid, "")
                    comment = "--"
                    if len(comment) > 80:
                        comment = comment[:77] + "..."
                    f.write(f"{r+1:4d} | {nid:11s} | {dist:7.4f} | {fmt_pident(pid):14s} | {in_blast:15s} | {comment}\n")
                f.write("\n")

            f.write("\n" + "="*90 + "\n\n")

        f.write("Final search statistics\n")
        print("\nFinal search statistics")
        f.write("-"*70 + "\n")
        print("-"*70)
        f.write("Method        | Average QPS | Average Recall@N\n")
        print("Method        | Average QPS | Average Recall@N")
        f.write("-"*70 + "\n")
        print("-"*70)
        for m in methods:
            label = ANN_METHODS[m]
            qps_av = stats[m].qps / len(q_ids)
            recall_av = stats[m].recall / len(q_ids)
            f.write(f"{label:13s} | {qps_av:11.4f} | {recall_av:16.2f}\n")
            print(f"{label:13s} | {qps_av:11.4f} | {recall_av:16.2f}")
        # BLAST baseline in final stats (Average QPS / Average Recall)
        if args.blast_tsv:
            blast_qps_str = f"{blast_qps:11.4f}" if blast_qps is not None else f"{'--':>11}"
            blast_rec_avg = (blast_ref_sum / float(len(q_ids))) if len(q_ids) > 0 else 0.0
            f.write(f"{'BLAST (Ref)':13s} | {blast_qps_str} | {blast_rec_avg:16.2f}\n")
            print(f"{'BLAST (Ref)':13s} | {blast_qps_str} | {blast_rec_avg:16.2f}")
        f.write("-"*70 + "\n")
        print("-"*70 + "\n")


    print(f"[Report] {args.output}")
    print(f"[Cache] {cache_dir}")


if __name__ == "__main__":
    main()
