# CosmoCluster Phone-First Experiment Loop (Phase 1)

This repository now includes a **phone-first ML experiment loop foundation**. Phase 1 focuses on a complete dummy end-to-end workflow so you can control experiments from GitHub (including GitHub Mobile on iPhone), while compute/orchestration happens via GitHub Actions.

## What this implements

- YAML experiment config + schema validation
- Deterministic run identity (`timestamp_sha_experiment_seed`)
- Standardized run outputs under `outputs/runs/<RUN_ID>/`
- Dummy run execution (no real model training yet)
- Mobile-first HTML run report generation
- Comparison report generation across runs
- GitHub Actions workflows for CI, run execution, and Pages publishing

## Architecture (Phase 1)

- **Control plane:** GitHub PRs + GitHub Actions + GitHub Mobile
- **Compute plane:** `LocalRunner` dummy backend (extensible to Modal/GCP)
- **Artifact plane:** Actions artifacts + GitHub Pages content

## Repo structure

- `experiments/` experiment registry, configs, and schema
- `src/` run launcher, report/comparison generation, utility modules
- `runners/` backend abstraction (`Runner`, `LocalRunner`, placeholders for Modal/GCP)
- `reports/` HTML templates and shared styles
- `scripts/` CLI entrypoints for validation and packaging
- `.github/workflows/` CI + run + Pages workflows

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/validate_config.py experiments/configs/example_baseline.yaml
python src/launch_run.py --config experiments/configs/example_baseline.yaml --backend local --notes "local smoke"
```

After a run, open:
- `outputs/runs/<RUN_ID>/report/index.html`

## Triggering from GitHub Actions (phone-friendly)

1. Merge to `main`.
2. In GitHub: **Actions → run-experiment → Run workflow**.
3. Set `config_path` and `backend`.
4. Monitor status from GitHub Mobile.
5. Download artifacts from the run page and/or open Pages report.

## GitHub setup checklist

1. Enable Actions for the repository.
2. In **Settings → Actions → General**, set workflow permissions as needed (recommend read/write for this foundation).
3. Add repo secrets/variables in **Settings → Secrets and variables → Actions**:
   - Secrets (optional for Phase 1, needed for future backends):
     - `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
     - `GCP_PROJECT_ID`, `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT`
     - `OPENAI_API_KEY` (only if you add LLM summaries)
   - Variables:
     - `DEFAULT_BACKEND` (e.g. `local`)
     - `PYTHON_VERSION` (e.g. `3.11`)
     - `REPORT_SITE_BASE_URL` (optional)
4. In **Settings → Pages**, set source to **GitHub Actions**.

## Standard run contract

Each run writes:

```text
outputs/runs/<RUN_ID>/
  manifest.json
  run_summary.md
  metrics.csv
  metrics.json
  config_frozen.yaml
  logs/
    train.log
    eval.log
  figures/
    loss_curve.png
    confusion_matrix.png
  plotly/
    training_curves.html
    embedding_projection.html
  report/
    index.html
  checkpoints/
    best.ckpt
```

## Compare runs

```bash
python src/compare_runs.py --run-dirs outputs/runs/<RUN_1> outputs/runs/<RUN_2>
```

Generates:

- `outputs/comparisons/<COMPARE_ID>/index.html`

## Adding a new backend later

Implement `Runner` methods in `runners/<new>_runner.py`:
- `submit(run_spec)`
- `status(job_id)`
- `wait(job_id)`
- `collect(job_id, output_dir)`

Then wire selection in `src/launch_run.py`.
