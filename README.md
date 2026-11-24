# UAD-JAGA

**Honest early-warning models for student graduation risk, quantified through a measurement-forward leakage audit.**

This repository holds the reproducible artifacts behind the manuscript
*UAD-JAGA: Honest early-warning models for student graduation risk, quantified through a
measurement-forward leakage audit* (submitted line of research: student-dropout early warning).

Every number in the manuscript is produced by code in this repository and stored in
`outputs/`; `analysis/verify_provenance.py` proves the stored numbers are internally
consistent without refitting. `analysis/rerun_pipeline.py` regenerates them from the raw data.

## Verified headline numbers

See `outputs/leakage_audit_v2_REAL.csv`, `outputs/cv_benchmark_v2_REAL.json`, and
`outputs/temporal_validation_v2_REAL.json`:

- **UMBJM cohort rebuilt from the 2019–2024 registers**: 4,497 students from the graduation
  register (4,134) and the exit/DO register (411), linked to a 238,779-row transcript with
  relative-term indexing; outcome = not-on-time-graduation (25.8% at-risk); non-academic
  exits excluded (Mutasi 41, Meninggal 6).
- **Measurement-forward fidelity audit** (XGBoost, 5-fold stratified CV, pooled OOF):
  semester-1-only `0.796` → year-1 cumulative `0.883` (**Δ +0.087**, measured early-term
  information — reported as a fidelity gain, not hidden leakage).
- **Public UCI benchmark**: semester-2 content inflates AUC `0.958 → 0.938` (−0.019).
- **Early-warning benchmark** (pooled OOF): UMBJM semester-1-only XGBoost F1 `0.462`,
  AUC `0.796`; year-1 F1 `0.670`, AUC `0.883`; UCI XGBoost F1 `0.849`, AUC `0.938`.
  Fold-level mean±std (ddof=1) is in the JSON artifact.
- **Temporal (cohort) generalisation** (train ≤ k−1 → test k, semesters 2021–2024):
  AUC range `0.676–0.839`; the sem1-only 2024 dip (0.676) is flagged as class-maturity
  structure of the immature admission cohort, not model failure.
- **Deployment metrics**: precision@top-100 `1.00` (UCI) / `0.99` (UMBJM semester-1;
  `1.00` year-1); TPR @ FPR ≤ 5% `0.787` (UCI) in legacy audit.
- **Calibration** (cohort-holdout 2023, year-1 features): raw Brier `0.109` — isotonic
  recalibration does not improve (`0.112`) → raw probabilities deployed.
- **ADASYN**: no AUC gain on any dataset (0.938→0.938; 0.796→0.796; 0.883→0.877).
- **Exact Tree SHAP** (XGBoost `pred_contribs`, risk-oriented logit; reconstruction verified
  ≤ 2e-3): driver case studies in `outputs/figures/shap_waterfall_*_REAL.png`; leading risk
  drivers on the v2 cohort: first-term grade indexes (`ips1`, cumulative `ips12`) and the
  count of C/D/E grades.
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
  Canonical cohort: 4,497 students (admission 2019–2024) built from the graduation
  register, the exit/DO register, and the aggregated transcript; direct identifiers
  (registration number, student name) were removed at export, and **case-level outputs
  contain no student IDs**. The raw cohort **is not redistributed** (privacy);
  anonymised extracts are available from the corresponding author upon formal request.

## Citation

If you use UAD-JAGA (protocol, data splits, or app) in your work, please cite the manuscript
when available. See `CITATION.cff`.
