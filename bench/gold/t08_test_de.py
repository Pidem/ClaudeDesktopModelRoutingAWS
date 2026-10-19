"""Hidden pytest suite for T08. The grader drops the model's solution.py and
counts.csv next to this file and runs `pytest -q`. Tests the pinned method:
log2FC = mean(log2(trt+1)) - mean(log2(ctrl+1)); p = Welch-free two-sample
t-test (equal_var=True) on log2(counts+1); padj = Benjamini-Hochberg."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import solution  # the model's file

HERE = pathlib.Path(__file__).resolve().parent
COUNTS = str(HERE / "counts.csv")
TRUE_UP = [f"gene_{i:04d}" for i in range(0, 10)]
TRUE_DOWN = [f"gene_{i:04d}" for i in range(10, 20)]


@pytest.fixture(scope="module")
def result():
    return solution.differential_expression(COUNTS)


@pytest.fixture(scope="module")
def reference():
    df = pd.read_csv(COUNTS, index_col=0)
    ctrl = [c for c in df.columns if c.startswith("ctrl")]
    trt = [c for c in df.columns if c.startswith("trt")]
    lc, lt = np.log2(df[ctrl] + 1), np.log2(df[trt] + 1)
    l2fc = lt.mean(axis=1) - lc.mean(axis=1)
    _, p = stats.ttest_ind(lt, lc, axis=1, equal_var=True)
    padj = stats.false_discovery_control(p, method="bh")
    return pd.DataFrame({"log2fc": l2fc, "pvalue": p, "padj": padj}, index=df.index)


def test_shape_and_columns(result):
    assert len(result) == 200
    for col in ("log2fc", "pvalue", "padj"):
        assert col in result.columns


def test_index_is_genes(result):
    assert "gene_0000" in result.index
    assert "gene_0199" in result.index


def test_up_genes_positive_fc(result):
    assert (result.loc[TRUE_UP, "log2fc"] > 1.0).all()


def test_down_genes_negative_fc(result):
    assert (result.loc[TRUE_DOWN, "log2fc"] < -1.0).all()


def test_log2fc_matches_reference(result, reference):
    diff = (result.loc[reference.index, "log2fc"] - reference["log2fc"]).abs()
    assert diff.max() < 0.05


def test_pvalue_matches_reference(result, reference):
    diff = (result.loc[reference.index, "pvalue"] - reference["pvalue"]).abs()
    assert diff.max() < 1e-6


def test_significant_set_recovers_true_de(result):
    sig = set(result.index[result["padj"] < 0.05])
    true_de = set(TRUE_UP + TRUE_DOWN)
    assert len(true_de & sig) >= 18            # recover almost all true DE genes


def test_padj_ge_pvalue(result):
    assert (result["padj"] >= result["pvalue"] - 1e-9).all()
