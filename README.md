# Protein Embedding ANN Search

Fast similarity search for protein sequences using ESM-2 embeddings and Approximate Nearest Neighbor (ANN) methods.

## Overview

This project implements efficient protein retrieval in embedding space using [ESM-2](https://github.com/facebookresearch/esm) protein language models, comparing multiple ANN algorithms (LSH, Hypercube, IVF-Flat, IVF-PQ, Neural LSH) against BLAST baseline for detecting **remote homologs**—proteins with low sequence identity (&lt;30%) but conserved structure/function.

## Key Features

- **Embeddings**: ESM-2 (t6_8M_UR50D) with mean pooling → 320-dim vectors
- **ANN Methods**: LSH, Hypercube LSH, IVF-Flat, IVF-PQ, Neural LSH
- **Evaluation**: Recall@N vs BLAST Top-N, QPS (queries/second)
- **Biological Validation**: UniProt/Pfam/GO annotation overlap for remote homology detection

## Quick Start

```bash
# 1. Generate embeddings
python protein_embed.py \
  -i data/swissprot.fasta \
  -o data/protein_vectors.dat

# 2. Run search & evaluation
python protein_search.py \
  -d data/protein_vectors.dat \
  -q data/targets.fasta \
  --method all \
  --run_blast --db_fasta data/swissprot.fasta

## Dependencies
Python 3.8+, PyTorch, fair-esm
NCBI BLAST+ (for ground truth)
NumPy, scikit-learn, tqdm

## Authors
Ioannis Petrakis
Ioannis Nikolopoulos
