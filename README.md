# UAD-JAGA

**Honest early-warning models for student graduation risk, quantified through a measurement-forward leakage audit.**

This repository holds the reproducible artifacts behind the manuscript
*UAD-JAGA: Honest early-warning models for student graduation risk, quantified through a
measurement-forward leakage audit* (submitted line of research: student-dropout early warning).

Every number in the manuscript is produced by code in this repository and stored in
`outputs/`; `analysis/verify_provenance.py` proves the stored numbers are internally
consistent without refitting. `analysis/rerun_pipeline.py` regenerates them from the raw data.

## Verified headline numbers

See `outputs/leakage_audit_v3_REAL.csv`, `outputs/cv_benchmark_v3_REAL.json`, and
`outputs/temporal_validation_v3_REAL.json`:

- **UMBJM cohort rebuilt from the 2019–2024 registers, restricted to the data contract of the
  source extract**: Reguler-admission SARJANA (S1) students only (RPL/transfers and D3/D4/
  professional/master populations have different nominal study durations and are excluded).
  Study universe **n = 1,928**; at-risk (not-on-time) **427 = 22.1%**; non-academic exits
  excluded (Mutasi 41, Meninggal 6); student identifiers removed (the published canonical
  table carries no NIM/name columns).
- **Measurement-forward fidelity audit** (XGBoost, 5-fold stratified CV, pooled OOF):
  semester-1-only `0.745` → year-1 cumulative `0.848` (**Δ +0.103**, measured early-term
  information — reported as a fidelity gain, not hidden leakage).
- **Public UCI benchmark**: semester-2 content inflates AUC `0.958 → 0.938` (−0.019).
- **Early-warning benchmark** (pooled OOF): UMBJM semester-1-only XGBoost F1 `0.503`,
  AUC `0.745`; year-1 F1 `0.674`, AUC `0.848`; UCI XGBoost F1 `0.849`, AUC `0.938`.
  Fold-level mean±std (ddof=1) is in the JSON artifact.
- **Temporal (cohort) generalisation** (train ≤ k−1 → test k): on the **mature** windows —
  where the on-time graduation clock has fully elapsed — year-1 AUC `0.833` (2021) and
  `0.902` (2022); semester-1 `0.699`/`0.714`. The still-immature 2023 (n = 35; 97.1% flagged)
  and 2024 (n = 6; 100% flagged) cohorts are reported but excluded from the headline range.
- **Deployment metrics**: precision@top-100 `1.00` (UCI) / `0.96` (UMBJM semester-1;
  `1.00` year-1); TPR @ FPR ≤ 5% `0.412` (UMBJM semester-1) / `0.600` (year-1).
- **Calibration** (cohort-holdout 2022, year-1 features): raw Brier `0.080` vs isotonic
  `0.076` — isotonic barely improves Brier but degrades ranking (AUC `0.902 → 0.889`), so raw
  probabilities are deployed and both variants are reported.
- **ADASYN**: no AUC gain anywhere (0.938→0.938; 0.745→0.737; 0.848→0.841 — slightly worse
  on the internal cohort).
- **Exact Tree SHAP** (XGBoost `pred_contribs`, risk-oriented logit; reconstruction verified
  ≤ 2e-3): driver case studies in `outputs/figures/shap_waterfall_*_REAL.png` and
  `outputs/figure4_shap_v3_REAL.png`; leading risk drivers on the v3 cohort: year-1 course
  load, first-term/cumulative grade indexes, and the count of C/D/E grades.
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
  Canonical cohort: 1,928 Reguler-admission SARJANA (S1) students (admission 2019–2024) built from the graduation
  register, the exit/DO register, and the aggregated transcript; direct identifiers
  (registration number, student name) were removed at export, and **case-level outputs
  contain no student IDs**. The raw cohort **is not redistributed** (privacy);
  anonymised extracts are available from the corresponding author upon formal request.

## Citation

If you use UAD-JAGA (protocol, data splits, or app) in your work, please cite the manuscript
when available. See `CITATION.cff`.
