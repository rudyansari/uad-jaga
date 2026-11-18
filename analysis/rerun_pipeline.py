#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAD-JAGA — audit + benchmark pipeline (verified protocol).

Reproduces:  outputs/cv_benchmark_REAL.json + outputs/tables/leakage_audit_REAL.csv

Datasets are NOT bundled (privacy + size). Provide them via environment:
    UCI_CSV=/path/to/uci.csv        # public benchmark ("Predict students' dropout and academic success")
    UMBJM_CSV=/path/to/umbjm.csv    # internal cohort (anonymised extract; see README "Data provenance")

Conventions: 5-fold stratified CV (seed 42), pooled out-of-fold metrics + fold-level
mean±std (ddof=1), positive class = at-risk (label 0), MinMax fitted per train fold,
ADASYN applied inside train folds only. All numbers land in the JSON artifact (no hand edits).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', 'outputs'))
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
RS = 42

UCI_CSV = os.environ.get('UCI_CSV')
UMBJM_CSV = os.environ.get('UMBJM_CSV')
if not (UCI_CSV and UMBJM_CSV and os.path.exists(UCI_CSV) and os.path.exists(UMBJM_CSV)):
    sys.exit("Set UCI_CSV and UMBJM_CSV environment variables to the raw dataset file paths (see README).")

MODELS = {
    'LogReg': LogisticRegression(max_iter=2000, random_state=RS),
    'RF': RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RS, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric='logloss', random_state=RS),
}


def metrics_at(y_true, proba0, thr=0.5, pos=0):
    pred = np.where(proba0 >= thr, pos, 1 - pos)
    return dict(precision=precision_score(y_true, pred, pos_label=pos, zero_division=0),
                recall=recall_score(y_true, pred, pos_label=pos, zero_division=0),
                f1=f1_score(y_true, pred, pos_label=pos, zero_division=0),
                accuracy=accuracy_score(y_true, pred),
                flagged_rate=float(pred[pred == pos].shape[0] / len(pred)))


def cv_one(X, y, model, use_adasyn=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    oof = np.zeros(len(y))
    per = []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        sc = MinMaxScaler().fit(X[tr])
        Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
        if use_adasyn:
            Xt, yt = ADASYN(random_state=RS).fit_resample(Xt, y[tr])
        else:
            yt = y[tr]
        m = clone(model)
        m.fit(Xt, yt)
        p0 = m.predict_proba(Xe)[:, 0]
        oof[te] = p0
        mm = metrics_at(y[te], p0)
        mm['auc'] = roc_auc_score((y[te] == 0).astype(int), p0)
        mm['fold'] = k
        per.append(mm)
    pooled = metrics_at(y, oof)
    pooled['auc'] = roc_auc_score((y == 0).astype(int), oof)
    p, r, t = precision_recall_curve((y == 0).astype(int), oof)
    f1c = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    thr_best = float(t[np.argmax(f1c)]) if len(t) else 0.5
    opt = metrics_at(y, oof, thr=thr_best)
    fpr, tpr, thr2 = roc_curve((y == 0).astype(int), oof)
    idx = np.where(fpr <= 0.05)[0]
    thr05 = float(thr2[idx[-1]]) if len(idx) else 1.0
    ops05 = metrics_at(y, oof, thr=thr05)
    ops05['fpr'] = float(fpr[idx[-1]]) if len(idx) else None
    ops05['tpr'] = float(tpr[idx[-1]]) if len(idx) else None
    sel = y[np.argsort(-oof)[:100]]
    pk = float((sel == 0).mean())

    def ms(key):
        v = np.array([f[key] for f in per])
        return (float(v.mean()), float(v.std(ddof=1)))

    st = {k: ms(k) for k in ['precision', 'recall', 'f1', 'accuracy', 'auc']}
    return pooled, st, per, oof, thr_best, opt, ops05, pk


def main():
    dfu = pd.read_csv(UCI_CSV, sep=';')
    dfu.columns = [c.replace('\t', '').strip() for c in dfu.columns]
    yu = dfu['Target'].astype(int).values
    fu_all = [c for c in dfu.columns if c != 'Target']
    fu_s1 = [c for c in fu_all if '2nd sem' not in c]
    Xu_all = dfu[fu_all].astype(float).values
    Xu_s1 = dfu[fu_s1].astype(float).values

    dfm = pd.read_csv(UMBJM_CSV, sep=';')
    ym = dfm['Target'].astype(int).values
    fm_all = [c for c in dfm.columns if c not in ('Target', 'NIM', 'Nama')]
    fm_s1 = [c for c in fm_all if 'Semester 2' not in c]
    Xm_all = dfm[fm_all].astype(float).values
    Xm_s1 = dfm[fm_s1].astype(float).values

    results = {}
    for dn, gname, X, y in [('UCI', 'Sem1-only', Xu_s1, yu), ('UCI', 'Full-36', Xu_all, yu),
                            ('UMBJM', 'Sem1-only', Xm_s1, ym), ('UMBJM', 'Full-12', Xm_all, ym)]:
        results.setdefault(dn, {})[gname] = {'n': int(len(y)), 'd': int(X.shape[1]),
                                             'rate0': float((y == 0).mean()), 'models': {}}
        for mname, m in list(MODELS.items()):
            pooled, st, per, oof, thr, opt, ops05, pk = cv_one(X, y, m, False)
            results[dn][gname]['models'][mname] = {
                'pooled': {k: float(v) for k, v in pooled.items()},
                'fold_mean_std': {k: [float(a), float(b)] for k, (a, b) in st.items()},
                'f1opt': {'thr': thr, **{k: float(v) for k, v in opt.items()}},
                'fpr05': {k: float(v) for k, v in ops05.items() if v is not None},
                'precision_at_100': pk,
                'oof_proba': oof.tolist()}
        pooled, st, per, oof, thr, opt, ops05, pk = cv_one(X, y, MODELS['XGBoost'], True)
        results[dn][gname]['models']['XGB+ADASYN'] = {
            'pooled': {k: float(v) for k, v in pooled.items()},
            'fold_mean_std': {k: [float(a), float(b)] for k, (a, b) in st.items()},
            'f1opt': {'thr': thr, **{k: float(v) for k, v in opt.items()}},
            'fpr05': {k: float(v) for k, v in ops05.items() if v is not None},
            'precision_at_100': pk,
            'oof_proba': oof.tolist()}

    with open(f'{OUT}/cv_benchmark_REAL.json', 'w') as f:
        json.dump(results, f, indent=2)

    rows = []
    for dn, gname, full in [('UCI', 'Sem1-only', 'Full-36'), ('UMBJM', 'Sem1-only', 'Full-12')]:
        auc_full = results[dn][full]['models']['XGBoost']['pooled']['auc']
        auc_s1 = results[dn][gname]['models']['XGBoost']['pooled']['auc']
        rows.append({'dataset': dn, 'features': full, 'model': 'XGBoost',
                     'auc': round(auc_full, 3), 'delta_vs_full': ''})
        rows.append({'dataset': dn, 'features': gname, 'model': 'XGBoost',
                     'auc': round(auc_s1, 3), 'delta_vs_full': round(auc_full - auc_s1, 3)})
    pd.DataFrame(rows).to_csv(f'{OUT}/tables/leakage_audit_REAL.csv', index=False)
    print("Wrote cv_benchmark_REAL.json + leakage_audit_REAL.csv")
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
