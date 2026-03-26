# CosmoCluster Phone-First Experiment Loop (Phase 1 + Phase 2)

This repository now includes a **phone-first ML experiment loop foundation**. Phase 1 established the end-to-end workflow, and Phase 2 extends the mobile UX with a richer GitHub Pages run dashboard for quick triage from GitHub Mobile on iPhone.


## Roadmap status

- ✅ **Phase 1:** deterministic run contract, dummy execution pipeline, run/comparison reports, CI wiring
- ✅ **Phase 2:** mobile-first Pages index with status badges, backend metadata, and top metrics on each run card
- 🚧 **Phase 3 (in progress):** Modal-first backend defaults, queueing/retries, and richer experiment controls

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
- **Compute plane:** `ModalRunner` starter backend (with local artifact-compatible collection), plus Local/GCP options
- **Artifact plane:** Actions artifacts + GitHub Pages content

## Repo structure

- `experiments/` experiment registry, configs, and schema
- `src/` run launcher, report/comparison generation, utility modules
- `runners/` backend abstraction (`Runner`, `LocalRunner`, `ModalRunner`, placeholder for GCP)
- `reports/` HTML templates and shared styles
- `scripts/` CLI entrypoints for validation and packaging
- `.github/workflows/` CI + run + Pages workflows

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/validate_config.py experiments/configs/example_baseline.yaml
python src/launch_run.py --config experiments/configs/example_baseline.yaml --backend modal --notes "modal smoke"
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
     - `DEFAULT_BACKEND` (e.g. `modal`)
     - `PYTHON_VERSION` (e.g. `3.11`)
     - `REPORT_SITE_BASE_URL` (optional)
4. In **Settings → Pages**, set source to **GitHub Actions**.


### Troubleshooting GitHub Pages 404 in CI

If `publish-pages` fails with `HttpError: Not Found` on a URL like `.../rest/pages/pages#get-a-...-pages-site`, the repo usually does not have Pages enabled yet.

1. Open **Settings → Pages** in your repository.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Ensure Actions can write Pages artifacts: **Settings → Actions → General → Workflow permissions → Read and write permissions**.
4. Re-run **Actions → publish-pages**.

The workflow includes a preflight check that fails early with these exact steps if Pages is missing.

### Node 20 deprecation note (GitHub Actions)

This repo now pins newer action majors that run on current GitHub-hosted runners (Node.js 24-compatible), and removes the temporary `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` override.
If you still see a deprecation warning, sync your branch and confirm your workflow run is using:

- `actions/checkout@v5`
- `actions/setup-python@v6`
- `actions/github-script@v8`
- `actions/configure-pages@v5`
- `actions/upload-artifact@v6`

## Modal setup (account, token, and GitHub secrets)

If the Modal website setup path is unclear, follow this exact flow.

### 1) Create / access your Modal account

1. Go to https://modal.com and sign in.
2. Create or select your workspace.

### 2) Create an API token in Modal

If you don't see token fields in your current settings page, Modal's UI may be on a different navigation layout for your workspace/account.

Use either path:

1. In Modal, switch to the correct workspace first.
2. Open the tokens page directly: `https://modal.com/settings/tokens`.
3. If the page still doesn't show token creation, use the CLI flow below to mint/set a token.
4. Copy both values shown by Modal:
   - **Token ID**
   - **Token Secret**

> Keep the token secret private. You may not be able to view it again after leaving the page.

### 3) Configure local credentials (optional but recommended)

From your laptop/desktop terminal:

```bash
pip install modal
# Interactive flow: creates a token and writes local profile config.
modal token new

# Explicit flow if you already have values:
modal token set --token-id <YOUR_MODAL_TOKEN_ID> --token-secret <YOUR_MODAL_TOKEN_SECRET>
```

Or export env vars for one shell session:

```bash
export MODAL_TOKEN_ID=<YOUR_MODAL_TOKEN_ID>
export MODAL_TOKEN_SECRET=<YOUR_MODAL_TOKEN_SECRET>
```

### 4) Configure GitHub Actions credentials (for workflow runs)

In your repo, open **Settings → Secrets and variables → Actions → New repository secret** and add:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Then run **Actions → run-experiment** with `backend=modal`.

### 5) Verify credentials are picked up

- In run artifacts, inspect `logs/modal_runner.log`.
- In the run directory, inspect `collect_metadata.json`.

If credentials are missing, this project intentionally falls back to simulated mode to preserve report generation and run artifacts.

## Modal-first usage notes (Phase 3 starter)

- The default example config (`experiments/configs/example_baseline.yaml`) uses `compute_target.backend: modal`.
- `ModalRunner` preserves the existing run artifact contract and writes Modal metadata to:
  - `collect_metadata.json`
  - `logs/modal_runner.log`
- Without credentials, the runner uses safe **simulated** mode while keeping reports/artifacts intact for iPhone triage.

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
