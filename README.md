# CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance

This package contains the CARE workflow and the utilities needed to construct and evaluate MIMIC-DOS from a locally licensed MIMIC-IV v3.1 source.


## Contents

- `python/care_workflow`: main CARE workflow.
- `python/agent_support`: shared fact registry and logging helpers.
- `python/care_common`: shared runtime helpers for CARE runs.
- `python/eval`: sample construction and feature export utilities.
- `sql/care_derived`: CARE-specific derived-table definitions used by the dataset builder.
- `prompts`: prompt templates used by CARE.

Rubric configuration files define the CARE category labels used by the workflow.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
mkdir -p data/samples data/feature_store data/manifests runs/logs
```

Configure `.env` with a loopback URL and model name for a locally hosted OpenAI-compatible LLM server, together with the remote advisory endpoint, model, and API key. CARE requires the local endpoint to use `localhost`, `127.0.0.1`, or `::1` and keeps it separate from the remote endpoint. The run scripts use `temperature=0.0`.

## Data

The reported evaluation sample and companion feature-store CSV are derived from restricted MIMIC-IV v3.1 records and are not included in this package. The package provides the complete cohort, label, and feature construction code so that credentialed MIMIC-IV users can build a local evaluation sample under the same task definition.

Set the licensed MIMIC-IV v3.1 and external `mimic-code` locations before building the database:

```bash
export MIMIC_RAW_CSV_DIR=/path/to/licensed/mimic-iv-csv-root
export MIMIC_CODE_DIR=/path/to/external/mimic-code-checkout
export MIMIC_DB_PATH=data/mimiciv.duckdb
```

This package does not vendor MIMIC-IV or `mimic-code`. `MIMIC_CODE_DIR` must point to a separate local checkout of the official MIT-LCP repository.

`python/build_mimic_db.py` ingests the licensed source, compiles the official concepts from the external `mimic-code` checkout, and then materializes the CARE-specific tables defined under `sql/care_derived`. For an existing DuckDB database that already contains the required official concepts, run `python/build_care_derived_tables.py` directly.

Build a local balanced 1,000-sample evaluation set and its feature package with:

```bash
python python/build_mimic_db.py

python python/eval/build_locked_scope_sample.py \
  --label-window 0_12 \
  --seed 74021 \
  --sample-size 1000 \
  --overlap-mode pair \
  --output data/samples/mimic_dos_local_y0_12_n1000_seed74021.csv \
  --manifest data/manifests/mimic_dos_local_y0_12_n1000_seed74021_manifest.json

python python/eval/export_locked_sample_package.py \
  --sample data/samples/mimic_dos_local_y0_12_n1000_seed74021.csv \
  --feature-output data/feature_store/mimic_dos_local_y0_12_n1000_seed74021__feature_package.csv \
  --manifest-output data/manifests/mimic_dos_local_y0_12_n1000_seed74021_feature_manifest.json
```

The reported locked sample also excludes evaluation pairs used in earlier development splits. Those identifier-bearing exclusion files are not distributed. The sample builder accepts one or more local `--exclude-csv` arguments when the same exclusion protocol is required. Without those files, the command above deterministically samples from the full eligible local pool and will not reproduce the reported sample fingerprint.

Set the generated local files for inference:

```bash
export SAMPLE_LOCK_FILE=data/samples/mimic_dos_local_y0_12_n1000_seed74021.csv
export FEATURE_STORE_CSV=data/feature_store/mimic_dos_local_y0_12_n1000_seed74021__feature_package.csv
export FEATURE_SOURCE=locked_csv
```

## Run

```bash
python python/care_workflow/run_care_inference.py --sample-size 1000
```

## Citation 

```bibtex
@misc{liu2026careprivacycompliantagenticreasoning,
      title={CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance}, 
      author={Haochen Liu and Weien Li and Rui Song and Zeyu Li and Chun Jason Xue and Xiao-Yang Liu and Sam Nallaperuma and Xue Liu and Ye Yuan},
      year={2026},
      eprint={2604.01113},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.01113}, 
}
```
