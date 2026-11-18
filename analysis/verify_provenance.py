#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAD-JAGA — provenance/self-consistency verifier.

Recomputes every headline number from the STORED out-of-fold probabilities inside
outputs/cv_benchmark_REAL.json and asserts they equal the stored metrics. If this script
passes without refitting, the published tables are internally consistent; reviewers can
re-run rerun_pipeline.py against the raw data for a full replication.

Usage:
    python analysis/verify_provenance.py              # needs nothing else
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, '..', 'outputs', 'cv_benchmark_REAL.json')


def metrics_from_oof(y_true, oof, thr=0.5, pos=0):
    pred = np.where(oof >= thr, pos, 1 - pos)
    return dict(precision=precision_score(y_true, pred, pos_label=pos, zero_division=0),
                recall=recall_score(y_true, pred, pos_label=pos, zero_division=0),
                f1=f1_score(y_true, pred, pos_label=pos, zero_division=0),
                accuracy=accuracy_score(y_true, pred),
                auc=roc_auc_score((np.asarray(y_true) == 0).astype(int), oof))


def main():
    bench = json.load(open(JSON_PATH))
    failures = []

    def truth_needed(ds):
        # truth labels are not stored in the artifact to keep it portable; we rely
        # on n + rate0 to reconstruct a *placeholder* only when UCI/UMBJM aren't
        # provided. For exactness we require env paths so truth is literal.
        path = os.environ.get('UCI_CSV' if ds == 'UCI' else 'UMBJM_CSV')
        if not path:
            return None
        import pandas as pd
        df = pd.read_csv(path, sep=';')
        return df['Target'].astype(int).values

    for ds in bench:
        y = truth_needed(ds)
        assert y is not None, f"Set env {'UCI_CSV' if ds=='UCI' else 'UMBJM_CSV'} to verify truth labels."
        for gname in bench[ds]:
            for mname, mv in bench[ds][gname]['models'].items():
                oof = np.array(mv['oof_proba'], dtype=float)
                recomputed = metrics_from_oof(y, oof)
                stored = mv['pooled']
                for key in ['precision', 'recall', 'f1', 'accuracy', 'auc']:
                    d = abs(recomputed[key] - stored[key])
                    if d > 1e-9:
                        failures.append((ds, gname, mname, key, stored[key], recomputed[key]))
                # ops metrics
                sel = y[np.argsort(-oof)[:100]]
                pk = float((sel == 0).mean())
                if abs(pk - mv['precision_at_100']) > 1e-9:
                    failures.append((ds, gname, mname, 'precision_at_100',
                                     mv['precision_at_100'], pk))
    if failures:
        print("MISMATCHES FOUND:")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print("PROVENANCE OK — every stored metric equals its recomputation from stored OOF probabilities.")


if __name__ == '__main__':
    main()
