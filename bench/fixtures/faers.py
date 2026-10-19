"""Seeded synthetic FAERS-like spontaneous-report tables in DuckDB for T06.

Task: per drug-event pair, compute the Proportional Reporting Ratio (PRR) and a
2x2 chi-square, using window/aggregate SQL. Signal is injected for one pair
(drugA x eventX) so a correct query surfaces it. Deterministic (seeded); the gold
query in bench/gold/t06_prr.sql defines the expected result set. Run:

    uv run --group bench python bench/fixtures/faers.py

Schema (one row per report-drug and report-event; a report may list several):
    reports(report_id)
    report_drug(report_id, drug)
    report_event(report_id, event)
Writes bench/fixtures/faers.duckdb.
"""
from __future__ import annotations

import pathlib
import random

import duckdb

OUT = pathlib.Path(__file__).resolve().parent / "faers.duckdb"
SEED = 42
N_REPORTS = 5000
DRUGS = ["drugA", "drugB", "drugC", "drugD"]
EVENTS = ["eventX", "eventY", "eventZ", "eventW"]


def build() -> None:
    rng = random.Random(SEED)
    if OUT.exists():
        OUT.unlink()
    con = duckdb.connect(str(OUT))
    con.execute("""
        CREATE TABLE reports (report_id INTEGER);
        CREATE TABLE report_drug (report_id INTEGER, drug VARCHAR);
        CREATE TABLE report_event (report_id INTEGER, event VARCHAR);
    """)
    reports, rd, re_ = [], [], []
    for rid in range(1, N_REPORTS + 1):
        reports.append((rid,))
        drugs = {d for d in DRUGS if rng.random() < 0.28}
        if not drugs:
            drugs = {rng.choice(DRUGS)}
        events = {e for e in EVENTS if rng.random() < 0.22}
        # Inject a disproportionate drugA -> eventX association.
        if "drugA" in drugs and rng.random() < 0.45:
            events.add("eventX")
        if not events:
            events = {rng.choice(EVENTS)}
        for d in drugs:
            rd.append((rid, d))
        for e in events:
            re_.append((rid, e))
    con.executemany("INSERT INTO reports VALUES (?)", reports)
    con.executemany("INSERT INTO report_drug VALUES (?,?)", rd)
    con.executemany("INSERT INTO report_event VALUES (?,?)", re_)
    con.close()
    print(f"wrote {OUT} — {len(reports)} reports, {len(rd)} drug rows, {len(re_)} event rows")


if __name__ == "__main__":
    build()
