-- CARE compatibility view of the official MIMIC-IV DuckDB SOFA concept.
-- Upstream: mimic-iv/concepts_duckdb/score/sofa.sql in MIT-LCP/mimic-code.

DROP TABLE IF EXISTS mimiciv_derived.sofa_hourly;
CREATE TABLE mimiciv_derived.sofa_hourly AS
SELECT
  stay_id,
  hr,
  respiration_24hours AS sofa_resp,
  coagulation_24hours AS sofa_coag,
  liver_24hours AS sofa_liver,
  cardiovascular_24hours AS sofa_cardiovascular,
  cns_24hours AS sofa_cns,
  renal_24hours AS sofa_renal,
  sofa_24hours AS sofa_total
FROM mimiciv_derived.sofa
WHERE hr >= 0;
