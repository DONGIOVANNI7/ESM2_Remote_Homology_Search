# Protein Embedding ANN Search

Fast similarity search for protein sequences using ESM-2 embeddings and Approximate Nearest Neighbor (ANN) methods.

## Overview

This project implements efficient protein retrieval in embedding space using [ESM-2](https://github.com/facebookresearch/esm) protein language models, comparing multiple ANN algorithms (LSH, Hypercube, IVF-Flat, IVF-PQ, Neural LSH) against BLAST baseline for detecting **remote homologs**—proteins with low sequence identity (<30%) but conserved structure/function.
For a more detailed analysis read the pdf reports!

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
```

## Performance Comparison

| Method     | Avg Recall@50 | QPS   | Best For           |
| ---------- | ------------- | ----- | ------------------ |
| Neural LSH | 0.35          | 1,648 | Accuracy           |
| IVF-Flat   | 0.33          | 2,146 | Balance            |
| IVF-PQ     | 0.32          | 2,839 | Speed/Memory       |
| Hypercube  | 0.21          | 3,291 | Maximum throughput |

## Dependencies

* **Python 3.8+**
* **PyTorch**
* **fair-esm**
* **NCBI BLAST+** (for ground truth)
* **NumPy**
* **scikit-learn**
* **tqdm**

## Authors

* **Ioannis Petrakis** * **Ioannis Nikolopoulos** ````

***

Would you like me to help you add any badges (like build status or Python version) to the top of the README?
