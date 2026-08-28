# CARE-derived tables

These SQL files define the project-specific tables used to construct MIMIC-DOS. They run after the official MIMIC-IV DuckDB concepts have been compiled from an external `mimic-code` checkout.

1. `01_sofa_hourly.sql` exposes the official hourly SOFA output under the compact column names used by CARE.
2. `02_occult_hypoperfusion_slice.sql` aggregates the hourly MAP, heart-rate, pain, and RASS measurements used by the cohort builder.
3. `03_sofa_labels_6_12.sql` retains the earlier T+6 through T+12 label option. The reported T+1 through T+12 outcome is computed directly from `sofa_hourly` in `python/eval/build_locked_scope_sample.py`.

All tables are materialized inside the user's local DuckDB database. No MIMIC-IV records are included in this repository.
