"""Reference solution for T08 — used only to validate the hidden test suite (a
correct implementation must pass all tests). The model under test supplies its
own solution.py with the same `differential_expression` contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def differential_expression(counts_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(counts_csv_path, index_col=0)
    ctrl = [c for c in df.columns if c.startswith("ctrl")]
    trt = [c for c in df.columns if c.startswith("trt")]
    log_ctrl = np.log2(df[ctrl].to_numpy() + 1.0)
    log_trt = np.log2(df[trt].to_numpy() + 1.0)
    log2fc = log_trt.mean(axis=1) - log_ctrl.mean(axis=1)
    _, pvalue = stats.ttest_ind(log_trt, log_ctrl, axis=1, equal_var=True)
    padj = stats.false_discovery_control(pvalue, method="bh")
    return pd.DataFrame({"log2fc": log2fc, "pvalue": pvalue, "padj": padj}, index=df.index)
