#!/usr/bin/env python3
"""
UAD-JAGA v3: rebuild UMBJM cohort restricted to Reguler + SARJANA (S1) students
and re-run the full verified pipeline (fidelity audit, benchmark, temporal,
calibration, SHAP, artifacts). All outputs *_v3_REAL.* .
Rationale (data contract): the source file covers Reguler S1 students only;
RPL/Pindahan and D3/Profesi/S2/D4 have different study durations and must be excluded.
"""
import json, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from imblearn.over_sampling import ADASYN
import joblib

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "Data")

# ---------------------------------------------------------------- cohort build
ds = pd.read_csv(os.path.join(HERE, "umbjm_v2_canonical.csv"))
kel = pd.read_csv(os.path.join(DATA, "kelulusan.csv"))
ext = pd.read_csv(os.path.join(DATA, "keluar_do_pindah.csv"))
reg = pd.concat([kel[["NIM", "JENJANG", "JENIS PENDAFTARAN"]],
                 ext[["NIM", "JENJANG", "JENIS PENDAFTARAN"]]])
ds["nim_str"] = ds["nim"].astype(str).str.replace(r"\.0$", "", regex=True)
reg["nim_str"] = reg["NIM"].astype(str)
m = ds.merge(reg.drop_duplicates("nim_str"), on="nim_str", how="left")
assert m["JENJANG"].notna().all()

cohort = m[(m["JENJANG"] == "SARJANA S1") & (m["jenis_daftar_code"] == 0)].copy()
cohort = cohort.drop(columns=["jenis_daftar_code", "JENJANG", "JENIS PENDAFTARAN"])
n = len(cohort); risk = int(cohort["outcome"].sum())
print(f"[cohort] Reguler+S1: n={n}  at-risk={risk} ({risk/n*100:.1f}%)  "
      f"angkatan={sorted(cohort['angkatan'].unique())}")
cohort.to_csv(os.path.join(HERE, "umbjm_v3_canonical.csv"), index=False)

FE_SEM1 = ["kode_prodi", "ips1", "n_mk_s1", "n_CDE_s1", "angka_mean_s1"]
FE_Y1 = FE_SEM1 + ["ips12", "n_mk_s12", "n_CDE_s12"]
y = cohort["outcome"].astype(int).values

def make_models(seed=7):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                                     subsample=0.9, colsample_bytree=0.9,
                                     eval_metric="logloss", random_state=seed),
    }

def eval_cv(X, y, name):
    skf = StratifiedKFold(5, shuffle=True, random_state=7)
    out = {}
    for mname, base in make_models().items():
        oof = np.zeros(len(y)); fold_auc = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = make_models()[mname].fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            oof[te] = p
            fold_auc.append(roc_auc_score(y[te], p))
        pred = (oof >= 0.5).astype(int)
        # ADASYN variant (same protocol, oversample inside train fold only)
        oof_a = np.zeros(len(y)); fold_auc_a = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr, ytr = ADASYN(random_state=7).fit_resample(sc.transform(X[tr]), y[tr])
            m = make_models()[mname].fit(Xtr, ytr)
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            oof_a[te] = p
            fold_auc_a.append(roc_auc_score(y[te], p))
        order = np.argsort(-oof)
        k100 = min(100, len(y))
        entry = {
            "pooled": {
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y, pred)),
                "auc": float(roc_auc_score(y, oof)),
            },
            "fold_mean_std": {"auc": [float(np.mean(fold_auc)), float(np.std(fold_auc, ddof=1))]},
            "precision_at_100": float(y[order[:k100]].mean()),
            "adasyn": {"auc": float(roc_auc_score(y, oof_a)),
                       "fold_auc": [float(np.mean(fold_auc_a)), float(np.std(fold_auc_a, ddof=1))]},
        }
        out[mname] = entry
        print(f"[cv {name}] {mname}: AUC {entry['pooled']['auc']:.3f} "
              f"F1 {entry['pooled']['f1']:.3f} P@100 {entry['precision_at_100']:.2f} "
              f"| ADASYN AUC {entry['adasyn']['auc']:.3f}")
    return out

X1 = cohort[FE_SEM1].astype(float).values
X2 = cohort[FE_Y1].astype(float).values
bench = {"Sem1-only": {"features": FE_SEM1, "models": eval_cv(X1, y, "Sem1")},
         "Year1-full": {"features": FE_Y1, "models": eval_cv(X2, y, "Year1")}}
bench["meta"] = {"n": n, "at_risk": risk, "rate": risk / n,
                 "positive_class": 1, "protocol": "5-fold stratified pooled OOF",
                 "cohort": "UMBJM Reguler + SARJANA S1, angkatan 2019-2024"}
json.dump(bench, open(os.path.join(HERE, "cv_benchmark_v3_REAL.json"), "w"), indent=1)

au1 = bench["Sem1-only"]["models"]["XGBoost"]["pooled"]["auc"]
au2 = bench["Year1-full"]["models"]["XGBoost"]["pooled"]["auc"]
pd.DataFrame([
    {"dataset": "UMBJM-v3-Reguler-S1", "features": "semester-1-only", "n": n, "auc": round(au1, 3)},
    {"dataset": "UMBJM-v3-Reguler-S1", "features": "year-1-cumulative", "n": n, "auc": round(au2, 3)},
    {"dataset": "UMBJM-v3-Reguler-S1", "features": "delta (fidelity gain)", "n": n, "auc": round(au2 - au1, 3)},
]).to_csv(os.path.join(HERE, "leakage_audit_v3_REAL.csv"), index=False)
print(f"[audit] fidelity delta = {au2-au1:+.3f}")

# ---------------------------------------------------------------- temporal
rows = []
coh = cohort.copy()
for k in [2021, 2022, 2023, 2024]:
    tr = coh["angkatan"] < k; te = coh["angkatan"] == k
    if te.sum() < 30 or int(y[tr.values].sum()) < 20:
        rows.append({"test_cohort": k, "n_test": int(te.sum()), "pos_test": int(y[te.values].sum()),
                     "auc_sem1": None, "auc_year1": None, "note": "insufficient"})
        continue
    r = {"test_cohort": k, "n_test": int(te.sum()), "pos_test": int(y[te.values].sum())}
    for col, FEx in [("auc_sem1", FE_SEM1), ("auc_year1", FE_Y1)]:
        sc = StandardScaler().fit(cohort.loc[tr, FEx].astype(float).values)
        mdl = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                                subsample=0.9, colsample_bytree=0.9,
                                eval_metric="logloss", random_state=7)
        mdl.fit(sc.transform(cohort.loc[tr, FEx].astype(float).values), y[tr.values])
        p = mdl.predict_proba(sc.transform(cohort.loc[te, FEx].astype(float).values))[:, 1]
        r[col] = round(float(roc_auc_score(y[te.values], p)), 3)
    rows.append(r)
temp = pd.DataFrame(rows)
temp.to_csv(os.path.join(HERE, "temporal_validation_v3_REAL.csv"), index=False)
json.dump({"rows": rows}, open(os.path.join(HERE, "temporal_validation_v3_REAL.json"), "w"), indent=1)
print(temp.to_string(index=False))

# ---------------------------------------------------------------- calibration (holdout 2023, year1)
tr = coh["angkatan"] <= 2022; te = coh["angkatan"] == 2023
Xtr = cohort.loc[tr, FE_Y1].astype(float).values; ytr = y[tr.values]
Xte = cohort.loc[te, FE_Y1].astype(float).values; yte = y[te.values]
sc = StandardScaler().fit(Xtr)
mdl = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=7)
mdl.fit(sc.transform(Xtr), ytr)
p_raw = mdl.predict_proba(sc.transform(Xte))[:, 1]
# isotonic fit on internal 85/15 split of train
rng = np.random.RandomState(7); idx = rng.permutation(len(ytr))
cut = int(0.85 * len(ytr)); itr, iva = idx[:cut], idx[cut:]
m2 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                       subsample=0.9, colsample_bytree=0.9,
                       eval_metric="logloss", random_state=7)
m2.fit(sc.transform(Xtr[itr]), ytr[itr])
p_val = m2.predict_proba(sc.transform(Xtr[iva]))[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, ytr[iva])
p_iso = iso.predict(p_raw)
cal = {"design": "cohort holdout 2023, year-1 features, isotonic fit on internal 85/15 split",
       "n_train": int(tr.sum()), "n_test": int(te.sum()),
       "brier_raw": float(brier_score_loss(yte, p_raw)),
       "brier_isotonic": float(brier_score_loss(yte, p_iso))}
json.dump(cal, open(os.path.join(HERE, "calibration_v3_REAL.json"), "w"), indent=1)
print(f"[calib] raw {cal['brier_raw']:.3f} vs isotonic {cal['brier_isotonic']:.3f}")

# ---------------------------------------------------------------- SHAP (year-1, real high-risk row)
sc_f = StandardScaler().fit(X2)
mF = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                       subsample=0.9, colsample_bytree=0.9,
                       eval_metric="logloss", random_state=7).fit(sc_f.transform(X2), y)
p_all = mF.predict_proba(sc_f.transform(X2))[:, 1]
i_risk = int(np.argmax(p_all)); i_safe = int(np.argmin(p_all))
def shap_row(i):
    xs = sc_f.transform(X2[i:i + 1])
    contrib = mF.get_booster().predict(xgb.DMatrix(xs, feature_names=FE_Y1), pred_contribs=True)[0]
    sv, base = contrib[:-1], float(contrib[-1])
    order = np.argsort(-np.abs(sv))
    return {"row_index": i, "risk": float(p_all[i]),
            "base_logit": base,
            "drivers": [{"feature": FE_Y1[j], "contribution": float(sv[j]),
                         "value": float(X2[i, j])} for j in order[:8]],
            "features": {f: float(v) for f, v in zip(FE_Y1, X2[i])}}
sh = {"case": shap_row(i_risk)}
json.dump(sh, open(os.path.join(HERE, "shap_umbjm_v3_REAL.json"), "w"), indent=1)
print(f"[shap] top driver: {sh['case']['drivers'][0]['feature']} "
      f"{sh['case']['drivers'][0]['contribution']:+.3f} (risk {p_all[i_risk]:.3f})")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = sh["case"]["drivers"][::-1]
plt.figure(figsize=(6, 3.4))
plt.barh([x["feature"] for x in d], [x["contribution"] for x in d],
         color=["#C0392B" if x["contribution"] > 0 else "#1F618D" for x in d])
plt.axvline(0, color="black", lw=.8)
plt.xlabel("SHAP contribution to at-risk log-odds")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "figure4_shap_v3_REAL.png"), dpi=200); plt.close()

t2 = temp.dropna(subset=["auc_year1"])
plt.figure(figsize=(6, 3.2))
plt.plot(t2["test_cohort"], t2["auc_sem1"], "o-", label="semester-1-only")
plt.plot(t2["test_cohort"], t2["auc_year1"], "s-", label="year-1 cumulative")
plt.ylim(0.5, 1.0); plt.xlabel("test admission cohort (train <= k-1)"); plt.ylabel("AUC")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(HERE, "figure3_temporal_v3_REAL.png"), dpi=200); plt.close()

# ---------------------------------------------------------------- artifacts
ART = os.path.join(HERE, "app_artifacts")
os.makedirs(ART, exist_ok=True)
for tag, FEx, Xx in [("umbjm_v3_sem1", FE_SEM1, X1), ("umbjm_v3_year1", FE_Y1, X2)]:
    scx = StandardScaler().fit(Xx)
    mx = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06,
                           subsample=0.9, colsample_bytree=0.9,
                           eval_metric="logloss", random_state=7).fit(scx.transform(Xx), y)
    joblib.dump(mx, os.path.join(ART, f"model_{tag}.joblib"))
    joblib.dump(scx, os.path.join(ART, f"scaler_{tag}.joblib"))
    joblib.dump({"features": FEx, "n_rows": n, "at_risk_rate": risk / n,
                 "pos_label": 1, "dataset": tag}, os.path.join(ART, f"spec_{tag}.joblib"))
    print(f"[artifact] {tag}: {len(FEx)} features")

# preset rows (real)
p1 = mF.predict_proba(sc_f.transform(X1 and X2)) if False else None
json.dump({"safe": {f: float(v) for f, v in zip(FE_Y1, X2[i_safe])},
           "risk": {f: float(v) for f, v in zip(FE_Y1, X2[i_risk])}},
          open(os.path.join(HERE, "real_preset_rows_v3.json"), "w"), indent=1)
print(f"[presets] safe p={p_all[i_safe]:.4f} risk p={p_all[i_risk]:.4f}")
print("\nV3 PIPELINE DONE")
