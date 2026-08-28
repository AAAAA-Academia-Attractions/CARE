-- Legacy T+6 through T+12 label table retained for the optional 6_12 setting.
-- The reported 0_12 task computes its label directly in the sample builder.

DROP TABLE IF EXISTS mimiciv_derived.sofa_labels;
CREATE TABLE mimiciv_derived.sofa_labels AS
WITH future_window AS (
    SELECT
        stay_id,
        hr AS t_base,
        sofa_total AS sofa_t,
        MAX(sofa_total) OVER (
            PARTITION BY stay_id
            ORDER BY hr
            ROWS BETWEEN 6 FOLLOWING AND 12 FOLLOWING
        ) AS sofa_future_max_6_12,
        COUNT(sofa_total) OVER (
            PARTITION BY stay_id
            ORDER BY hr
            ROWS BETWEEN 6 FOLLOWING AND 12 FOLLOWING
        ) AS valid_future_hours
    FROM mimiciv_derived.sofa_hourly
)
SELECT
    stay_id,
    t_base,
    sofa_t,
    sofa_future_max_6_12,
    sofa_future_max_6_12 - sofa_t AS delta_sofa_6_12,
    CASE
        WHEN sofa_t IS NULL THEN NULL
        WHEN valid_future_hours < 4 THEN NULL
        WHEN sofa_future_max_6_12 - sofa_t >= 2 THEN 1
        ELSE 0
    END AS y_deteriorate_delta2_6_12,
    valid_future_hours
FROM future_window
WHERE t_base >= 0;
