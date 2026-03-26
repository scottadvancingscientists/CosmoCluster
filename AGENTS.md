# AGENTS.md

## Agent operating rules for CosmoCluster

1. Never create a new experiment path without wiring it into report generation.
2. Never break the standard run output contract under `outputs/runs/<RUN_ID>/...`.
3. Always preserve logs and partial outputs on failure.
4. Prefer additive changes over repo-wide refactors.
5. Keep reports mobile-friendly (optimize for iPhone Safari).
6. Any new metric added to training/eval must also appear in:
   - `metrics.json`
   - `metrics.csv`
   - `report/index.html`
7. When adding a new chart:
   - create a static PNG thumbnail
   - optionally embed interactive Plotly in the report
8. When adding a new config:
   - validate against schema
   - provide a sensible default seed
   - ensure reproducibility metadata is captured
9. Update `README.md` when changing user-facing workflow.
10. Write code and docs assuming the user often operates from GitHub Mobile on iPhone.
