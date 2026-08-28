-- Hourly subjective and objective measurements used to construct MIMIC-DOS.
-- This table is materialized locally from licensed MIMIC-IV records.

DROP TABLE IF EXISTS mimiciv_derived.occult_hypoperfusion_slice;
CREATE TABLE mimiciv_derived.occult_hypoperfusion_slice AS
WITH dense_grid AS (
    SELECT
        sh.stay_id,
        sh.hr,
        ih.endtime AS t_eval,
        ih.endtime - INTERVAL '1' HOUR AS t_eval_minus_1h
    FROM mimiciv_derived.sofa_hourly sh
    JOIN mimiciv_derived.icustay_hourly ih
        ON sh.stay_id = ih.stay_id
       AND sh.hr = ih.hr
),
raw_maps AS (
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.valuenum AS map_val,
        CASE WHEN ce.itemid = 220052 THEN 1 ELSE 2 END AS source_rank
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid IN (220052, 220181)
      AND ce.valuenum IS NOT NULL
),
unique_maps AS (
    SELECT
        stay_id,
        charttime,
        map_val
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY stay_id, charttime
                ORDER BY source_rank ASC
            ) AS rn
        FROM raw_maps
    )
    WHERE rn = 1
),
map_spans AS (
    SELECT
        stay_id,
        charttime AS span_start,
        map_val,
        LEAST(
            charttime + INTERVAL '15' MINUTE,
            COALESCE(
                LEAD(charttime) OVER (PARTITION BY stay_id ORDER BY charttime),
                charttime + INTERVAL '15' MINUTE
            )
        ) AS span_end
    FROM unique_maps
),
obj_metrics AS (
    SELECT
        dg.stay_id,
        dg.hr,
        SUM(
            EXTRACT(EPOCH FROM (
                LEAST(ms.span_end, dg.t_eval)
                - GREATEST(ms.span_start, dg.t_eval_minus_1h)
            )) / 60.0
        ) AS map_covered_minutes_last1h,
        SUM(
            CASE WHEN ms.map_val < 60 THEN
                EXTRACT(EPOCH FROM (
                    LEAST(ms.span_end, dg.t_eval)
                    - GREATEST(ms.span_start, dg.t_eval_minus_1h)
                )) / 60.0
            ELSE 0 END
        ) AS map_low_minutes_last1h_thr60,
        SUM(
            CASE WHEN ms.map_val < 65 THEN
                EXTRACT(EPOCH FROM (
                    LEAST(ms.span_end, dg.t_eval)
                    - GREATEST(ms.span_start, dg.t_eval_minus_1h)
                )) / 60.0
            ELSE 0 END
        ) AS map_low_minutes_last1h_thr65
    FROM dense_grid dg
    LEFT JOIN map_spans ms
        ON dg.stay_id = ms.stay_id
       AND ms.span_start < dg.t_eval
       AND ms.span_end > dg.t_eval_minus_1h
    GROUP BY dg.stay_id, dg.hr
),
hr_metrics AS (
    SELECT
        dg.stay_id,
        dg.hr,
        MEDIAN(ce.valuenum) AS hr_median_last1h
    FROM dense_grid dg
    LEFT JOIN mimiciv_icu.chartevents ce
        ON dg.stay_id = ce.stay_id
       AND ce.itemid = 220045
       AND ce.charttime > dg.t_eval_minus_1h
       AND ce.charttime <= dg.t_eval
    GROUP BY dg.stay_id, dg.hr
),
subj_metrics AS (
    SELECT
        dg.stay_id,
        dg.hr,
        MAX(CASE WHEN ce.itemid = 223791 THEN ce.valuenum END) AS pain_max_last1h,
        COUNT(CASE WHEN ce.itemid = 228096 THEN 1 END) AS rass_n_last1h,
        MAX(CASE WHEN ce.itemid = 228096 THEN ce.valuenum END) AS rass_max_last1h,
        MIN(CASE WHEN ce.itemid = 228096 THEN ce.valuenum END) AS rass_min_last1h
    FROM dense_grid dg
    LEFT JOIN mimiciv_icu.chartevents ce
        ON dg.stay_id = ce.stay_id
       AND ce.itemid IN (223791, 228096)
       AND ce.charttime > dg.t_eval_minus_1h
       AND ce.charttime <= dg.t_eval
    GROUP BY dg.stay_id, dg.hr
)
SELECT
    dg.stay_id,
    dg.hr,
    CASE
        WHEN COALESCE(om.map_covered_minutes_last1h, 0) > 0
        THEN COALESCE(om.map_low_minutes_last1h_thr60, 0)
        ELSE NULL
    END AS map_low_minutes_last1h_thr60,
    CASE
        WHEN COALESCE(om.map_covered_minutes_last1h, 0) > 0
        THEN COALESCE(om.map_low_minutes_last1h_thr65, 0)
        ELSE NULL
    END AS map_low_minutes_last1h_thr65,
    CASE
        WHEN COALESCE(om.map_covered_minutes_last1h, 0) > 0 THEN TRUE
        ELSE FALSE
    END AS has_map_coverage_last1h,
    COALESCE(om.map_covered_minutes_last1h, 0) AS map_covered_minutes_last1h,
    hrm.hr_median_last1h,
    sm.pain_max_last1h,
    sm.rass_n_last1h,
    sm.rass_max_last1h,
    sm.rass_min_last1h,
    sm.pain_max_last1h AS pain_last_last1h,
    sm.rass_max_last1h AS rass_last_last1h,
    CASE
        WHEN COALESCE(om.map_low_minutes_last1h_thr60, 0) >= 30
         AND hrm.hr_median_last1h >= 110
         AND sm.pain_max_last1h IS NOT NULL
         AND sm.pain_max_last1h <= 1
         AND sm.rass_n_last1h >= 1
         AND sm.rass_max_last1h <= 1
         AND sm.rass_min_last1h >= -1
        THEN TRUE ELSE FALSE
    END AS occult_hypoperfusion_slice_thr60,
    CASE
        WHEN COALESCE(om.map_low_minutes_last1h_thr65, 0) >= 30
         AND hrm.hr_median_last1h >= 110
         AND sm.pain_max_last1h IS NOT NULL
         AND sm.pain_max_last1h <= 1
         AND sm.rass_n_last1h >= 1
         AND sm.rass_max_last1h <= 1
         AND sm.rass_min_last1h >= -1
        THEN TRUE ELSE FALSE
    END AS occult_hypoperfusion_slice_thr65
FROM dense_grid dg
LEFT JOIN obj_metrics om
    ON dg.stay_id = om.stay_id AND dg.hr = om.hr
LEFT JOIN hr_metrics hrm
    ON dg.stay_id = hrm.stay_id AND dg.hr = hrm.hr
LEFT JOIN subj_metrics sm
    ON dg.stay_id = sm.stay_id AND dg.hr = sm.hr;
