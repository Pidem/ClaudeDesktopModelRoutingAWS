-- Gold cohort for T02. Index (as-of) date = 2026-01-01.
-- Patients with Type 2 diabetes AND a metformin exposure, whose MOST RECENT
-- HbA1c in the 12 months before the index date is > 7.0 %. Returns person_id.
WITH last_hba1c AS (
    SELECT person_id, value_as_number,
           ROW_NUMBER() OVER (PARTITION BY person_id
                              ORDER BY measurement_date DESC) AS rn
    FROM measurement
    WHERE measurement_concept_id = 3004410
      AND measurement_date >= DATE '2025-01-01'
      AND measurement_date <  DATE '2026-01-01'
)
SELECT DISTINCT p.person_id
FROM person p
JOIN condition_occurrence c
  ON c.person_id = p.person_id AND c.condition_concept_id = 201826
JOIN drug_exposure d
  ON d.person_id = p.person_id AND d.drug_concept_id = 1503297
JOIN last_hba1c h
  ON h.person_id = p.person_id AND h.rn = 1 AND h.value_as_number > 7.0
ORDER BY p.person_id;
