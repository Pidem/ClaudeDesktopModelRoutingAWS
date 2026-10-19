"""Seeded synthetic RNA-seq count matrix for T08 (bioinformatics code).

Writes bench/fixtures/counts.csv: rows = genes (gene_0001 ..), columns = 8 samples
(ctrl_1..ctrl_4, trt_1..trt_4). A handful of genes carry a real fold change so a
correct differential-expression implementation flags them. The grader's hidden
pytest suite (bench/gold/t08_test_de.py) imports the model's `solution.py` and
checks its results against a reference computed the same way. Run:

    uv run --group bench python bench/fixtures/make_counts.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent / "counts.csv"
SEED = 7
N_GENES = 200
CTRL = [f"ctrl_{i}" for i in range(1, 5)]
TRT = [f"trt_{i}" for i in range(1, 5)]
# Genes 0-9 are truly up in treatment, 10-19 truly down; rest are noise.
TRUE_UP = list(range(0, 10))
TRUE_DOWN = list(range(10, 20))


def build() -> None:
    rng = np.random.default_rng(SEED)
    samples = CTRL + TRT
    data = {}
    genes = [f"gene_{i:04d}" for i in range(N_GENES)]
    base = rng.integers(50, 500, size=N_GENES).astype(float)
    for j, s in enumerate(samples):
        is_trt = s.startswith("trt")
        mult = np.ones(N_GENES)
        if is_trt:
            mult[TRUE_UP] = 4.0
            mult[TRUE_DOWN] = 0.25
        lam = base * mult
        data[s] = rng.poisson(lam).astype(int)
    df = pd.DataFrame(data, index=genes)
    df.index.name = "gene"
    df.to_csv(OUT)
    print(f"wrote {OUT} — {N_GENES} genes x {len(samples)} samples")


if __name__ == "__main__":
    build()
