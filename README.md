# UAD-JAGA

**Honest early-warning models for student graduation risk, quantified through a measurement-forward leakage audit.**

This repository holds the reproducible artifacts behind the manuscript
*UAD-JAGA: Honest early-warning models for student graduation risk, quantified through a
measurement-forward leakage audit* (submitted line of research: student-dropout early warning).

Every number in the manuscript is produced by code in this repository and stored in
`outputs/`; `analysis/verify_provenance.py` proves the stored numbers are internally
consistent without refitting. `analysis/rerun_pipeline.py` regenerates them from the raw data.

## Verified headline numbers

See `outputs/tables/leakage_audit_REAL.csv` and `outputs/cv_benchmark_REAL.json`:

- **Leakage audit** (XGBoost, 5-fold stratified CV, pooled OOF): UMBJM AUC `1.000 → 0.927`
  (full features → semester-1-only, −0.073); UCI `0.958 → 0.938` (−0.019).
- **Early-warning benchmark** (semester-1-only, pooled OOF): UCI XGBoost F1 `0.849`, AUC `0.938`;
  UMBJM XGBoost F1 `0.692`, AUC `0.927`. Fold-level mean±std (ddof=1) is in the JSON artifact.
- **Deployment metrics**: precision@top-100 `1.00` (UCI) / `0.98` (UMBJM); TPR @ FPR ≤ 5%
  `0.787` / `0.758`.
- **Calibration** (15% holdout): raw Brier `0.090` / `0.035`; isotonic recalibration does not
  improve (Brier `0.093` / `0.036`) → raw probabilities deployed.
- **ADASYN**: no AUC gain on either dataset (0.938→0.938; 0.927→0.928).
- **Exact Tree SHAP** (XGBoost `pred_contribs`, negated at-risk logit; reconstruction verified
  ≤ 2e-3): driver case studies in `outputs/figures/shap_waterfall_*_REAL.png`.
- **Human evaluation** (48 academic advisors, within-subject, 2,160 ratings): action-oriented
  narrative outscores SHAP-style and zero-shot GPT-4 formats on clarity (mean 4.61/5),
  actionability (4.58), trustworthiness (4.56); Wilcoxon p < 1e-15; Cliff’s δ 0.36–0.52
  (`outputs/tables/human_eval_descriptive_REAL.csv`, `human_eval_effects_REAL.csv`).

## Layout

```
analysis/
    rerun_pipeline.py       # audit + benchmark: regenerates cv_benchmark_REAL.json + leakage_audit_REAL.csv
    verify_provenance.py    # recomputes every stored metric from stored OOF probabilities (assert-equal)
app/
    app_uad_jaga.py         # Streamlit application (run: streamlit run app/app_uad_jaga.py)
    uad_jaga_core.py        # prediction/explanation core used by the app
    artifacts/              # trained models, scalers, and spec JSONs (semester-1-only features)
outputs/
    figures/                # verified figures used in the manuscript
    tables/                 # verified tables used in the manuscript
    *.json                  # machine-readable results (CV benchmark, calibration, SHAP, data claims)
docs/
requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt

# point to the raw dataset files (see "Data provenance" below) and regenerate:
UCI_CSV=/path/to/uci.csv UMBJM_CSV=/path/to/umbjm.csv python analysis/rerun_pipeline.py

# prove the stored numbers are self-consistent without refitting:
UCI_CSV=/path/to/uci.csv UMBJM_CSV=/path/to/umbjm.csv python analysis/verify_provenance.py

# run the application:
cd app && python -m streamlit run app_uad_jaga.py
```

## Data provenance

- **UCI public benchmark** — “Predict students’ dropout and academic success”
  (Cortez, 2014; DOI: 10.24432/C5TG7T). Analysis subset excludes the `Enrolled` class
  (binary Dropout vs Graduate; n = 3,630).
- **UMBJM internal cohort** — Universitas Muhammadiyah Banjarmasin, used under
  institutional permission letter No. **139/UM-BJM/R.1/II/2026** (23 February 2026).
  Direct identifiers (registration number, student name) were removed at export; the
  raw cohort **is not redistributed** (privacy). Anonymised extracts are available from
  the corresponding author upon formal request.

## Citation

If you use UAD-JAGA (protocol, data splits, or app) in your work, please cite the manuscript
when available. See `CITATION.cff`.
