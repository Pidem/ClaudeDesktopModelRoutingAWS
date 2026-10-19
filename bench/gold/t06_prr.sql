-- Gold for T06. Per drug-event pair, the 2x2 contingency over all reports:
--   a = reports with drug AND event, b = with drug not event,
--   c = without drug with event, d = neither.
-- PRR = (a/(a+b)) / (c/(c+d));  chi-square with Yates continuity correction.
-- Restrict to pairs with a >= 20 (stable signal). Rounded to 3 dp, ordered by PRR desc.
WITH n AS (SELECT COUNT(*)::DOUBLE AS total FROM reports),
drugs AS (SELECT DISTINCT drug FROM report_drug),
events AS (SELECT DISTINCT event FROM report_event),
pairs AS (SELECT drug, event FROM drugs CROSS JOIN events),
has_drug AS (SELECT DISTINCT report_id, drug FROM report_drug),
has_event AS (SELECT DISTINCT report_id, event FROM report_event),
cells AS (
    SELECT pr.drug, pr.event,
        (SELECT COUNT(*) FROM has_drug hd JOIN has_event he
             ON hd.report_id = he.report_id
             WHERE hd.drug = pr.drug AND he.event = pr.event)::DOUBLE AS a,
        (SELECT COUNT(DISTINCT report_id) FROM has_drug WHERE drug = pr.drug)::DOUBLE AS n_drug,
        (SELECT COUNT(DISTINCT report_id) FROM has_event WHERE event = pr.event)::DOUBLE AS n_event
    FROM pairs pr
),
contingency AS (
    SELECT drug, event, a,
           n_drug - a                    AS b,
           n_event - a                   AS c,
           (SELECT total FROM n) - n_drug - (n_event - a) AS d
    FROM cells
)
SELECT drug, event,
    ROUND((a / (a + b)) / (c / (c + d)), 3) AS prr,
    ROUND(
        ((a + b + c + d) * POWER(ABS(a * d - b * c) - (a + b + c + d) / 2.0, 2))
        / ((a + b) * (c + d) * (a + c) * (b + d)), 3) AS chi2
FROM contingency
WHERE a >= 20
ORDER BY prr DESC, drug, event;
