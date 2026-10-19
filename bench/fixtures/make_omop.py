"""Seeded synthetic OMOP CDM (subset) in DuckDB for T02 (cohort SQL).

Cohort the task asks for: patients with Type 2 diabetes (condition), on metformin
(drug_exposure), whose most recent HbA1c measurement in the 12 months before the
index date (2026-01-01) is > 7.0 %. Deterministic (seeded) so the gold SQL in
bench/gold/t02_cohort.sql returns a fixed row set. Run:

    uv run --group bench python bench/fixtures/make_omop.py

Writes bench/fixtures/omop.duckdb. Tables kept minimal but standards-shaped:
person, condition_occurrence, drug_exposure, measurement, concept.
"""
from __future__ import annotations

import pathlib
import random
from datetime import date, timedelta

import duckdb

OUT = pathlib.Path(__file__).resolve().parent / "omop.duckdb"
INDEX_DATE = date(2026, 1, 1)          # "as of" date the prompt gives the model
N_PERSONS = 400
SEED = 20260101

# Minimal concept ids (real OMOP vocab ids would come from Athena/CONCEPT).
C_T2D = 201826          # Type 2 diabetes mellitus (SNOMED-mapped standard)
C_METFORMIN = 1503297   # metformin ingredient
C_HBA1C = 3004410       # Hemoglobin A1c/Hemoglobin.total in Blood


def build() -> None:
    rng = random.Random(SEED)
    if OUT.exists():
        OUT.unlink()
    con = duckdb.connect(str(OUT))
    con.execute("""
        CREATE TABLE person (person_id INTEGER, gender_concept_id INTEGER,
                             year_of_birth INTEGER);
        CREATE TABLE condition_occurrence (condition_occurrence_id INTEGER,
                             person_id INTEGER, condition_concept_id INTEGER,
                             condition_start_date DATE);
        CREATE TABLE drug_exposure (drug_exposure_id INTEGER, person_id INTEGER,
                             drug_concept_id INTEGER, drug_exposure_start_date DATE);
        CREATE TABLE measurement (measurement_id INTEGER, person_id INTEGER,
                             measurement_concept_id INTEGER, measurement_date DATE,
                             value_as_number DOUBLE);
        CREATE TABLE concept (concept_id INTEGER, concept_name VARCHAR);
    """)
    con.executemany("INSERT INTO concept VALUES (?, ?)", [
        (C_T2D, "Type 2 diabetes mellitus"),
        (C_METFORMIN, "metformin"),
        (C_HBA1C, "Hemoglobin A1c"),
    ])

    persons, conds, drugs, meas = [], [], [], []
    cond_id = drug_id = meas_id = 0
    for pid in range(1, N_PERSONS + 1):
        persons.append((pid, rng.choice([8507, 8532]), rng.randint(1945, 1990)))
        has_t2d = rng.random() < 0.55
        on_metformin = has_t2d and rng.random() < 0.7
        if has_t2d:
            cond_id += 1
            conds.append((cond_id, pid, C_T2D,
                          INDEX_DATE - timedelta(days=rng.randint(400, 3000))))
        if on_metformin:
            drug_id += 1
            drugs.append((drug_id, pid, C_METFORMIN,
                          INDEX_DATE - timedelta(days=rng.randint(30, 900))))
        # Everyone may have HbA1c labs; 1-3 in the trailing ~18 months.
        for _ in range(rng.randint(0, 3)):
            meas_id += 1
            days_ago = rng.randint(1, 550)          # some fall outside the 12-mo window
            val = round(rng.uniform(5.4, 10.5), 1)
            meas.append((meas_id, pid, C_HBA1C, INDEX_DATE - timedelta(days=days_ago), val))

    con.executemany("INSERT INTO person VALUES (?,?,?)", persons)
    con.executemany("INSERT INTO condition_occurrence VALUES (?,?,?,?)", conds)
    con.executemany("INSERT INTO drug_exposure VALUES (?,?,?,?)", drugs)
    con.executemany("INSERT INTO measurement VALUES (?,?,?,?,?)", meas)
    con.close()
    print(f"wrote {OUT} — {len(persons)} persons, {len(conds)} T2D, "
          f"{len(drugs)} metformin, {len(meas)} HbA1c labs")


if __name__ == "__main__":
    build()
