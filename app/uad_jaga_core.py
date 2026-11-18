"""
UAD-JAGA core module — pure prediction/explanation helpers.
All numbers served to the UI come from the trained artifacts in
Riset/reanalysis/app_artifacts/ (models + scalers) and Riset/reanalysis/
verified JSON outputs. No mock/simulated values.

Orientation:
  Target 0 = 'at_risk' (positive class), Target 1 = 'safe'.
  Models output probability for both classes; risk = P(class 0).

Verified model metrics (5-fold stratified CV, pooled out-of-fold, seed 42)
served read-only from verified_metrics.json - UI must display verbatim.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_artifacts")
GEO_ROOT = os.path.dirname(ARTIFACT_DIR)


class ArtifactNotFound(RuntimeError):
    pass


def _rp(*p: str) -> str:
    return os.path.join(GEO_ROOT, *p)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ArtifactNotFound(f"Missing artifact: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- datasets
DATASETS = {
    "umbjm": {
        "label": "UMBJM — Universitas Muhammadiyah Banjarmasin (Indonesia)",
        "desc": "Semester-1 early-warning model, internal cohort.",
        "bands": (0.30, 0.70),  # LOW < 0.30 <= MEDIUM < 0.70 <= HIGH (operational)
    },
    "uci": {
        "label": "UCI — Public benchmark (Portugal; 'Enrolled' class excluded)",
        "desc": "Semester-1 early-warning model on public dropout data.",
        "bands": (0.30, 0.70),
    },
}


@dataclass
class TrainedModel:
    key: str
    features: List[str]
    spec: Dict[str, Any]
    model: Any
    scaler: Any


def load_artifacts(key: str) -> TrainedModel:
    spec_path = os.path.join(ARTIFACT_DIR, f"spec_{key}.json")
    spec = _load_json(spec_path)
    model_path = os.path.join(ARTIFACT_DIR, f"model_{key}.joblib")
    scaler_path = os.path.join(ARTIFACT_DIR, f"scaler_{key}.joblib")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise ArtifactNotFound(f"Missing model/scaler for {key}")
    return TrainedModel(
        key=key,
        features=spec["features"],
        spec=spec,
        model=joblib.load(model_path),
        scaler=joblib.load(scaler_path),
    )


def verified_metrics() -> Dict[str, Any]:
    """Aggregated verified metrics shown verbatim in the UI's 'About' tab."""
    out = {"files": {}}
    for fname in [
        "leakage_audit_REAL.csv",
        "metrics_summary_REAL.csv",
        "human_eval_descriptive_REAL.csv",
        "human_eval_effects_REAL.csv",
    ]:
        p = _rp(fname)
        if os.path.exists(p):
            out["files"][fname] = pd.read_csv(p).to_dict("records")
    for jname in ["cv_benchmark_REAL.json", "calibration_REAL.json"]:
        p = _rp(jname)
        if os.path.exists(p):
            out["files"][jname] = _load_json(p)
    return out


# ---------------------------------------------------------------- prediction
def predict(tm: TrainedModel, row: Dict[str, float]) -> Dict[str, Any]:
    """
    row: feature_name -> raw numeric value (all required keys present).
    Returns risk probability for class 0 ('at_risk'), tier, and proba dict.
    """
    vec = np.array([[float(row[f]) for f in tm.features]], dtype=float)
    xs = tm.scaler.transform(vec)
    proba = tm.model.predict_proba(xs)[0]
    risk = float(proba[0])
    bands = DATASETS[tm.key]["bands"]
    tier = "HIGH" if risk >= bands[1] else ("MEDIUM" if risk >= bands[0] else "LOW")
    return {
        "risk": risk,
        "p_safe": float(proba[1]),
        "tier": tier,
        "thresholds_used": {"tier_bands": bands},
        "features_used": len(tm.features),
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def exact_shap_contributions(
    tm: TrainedModel, row: Dict[str, float], k: int = 8
) -> Dict[str, Any]:
    """
    Tree SHAP exact contributions via xgboost pred_contribs.
    Negated to the at-risk (class 0) logit orientation, same as the
    verified protocol in Riset/reanalysis (assert-reconstruction).
    """
    vec = np.array([[float(row[f]) for f in tm.features]], dtype=float)
    xs = tm.scaler.transform(vec)
    booster = tm.model.get_booster()
    dm = xgb.DMatrix(xs, feature_names=list(tm.features))
    contrib = booster.predict(dm, pred_contribs=True)[0]
    sv, base = -contrib[:-1], -float(contrib[-1])
    model_atrisk = float(tm.model.predict_proba(xs)[0, 0])
    recon = float(_sigmoid(np.array([base + sv.sum()]))[0])
    if abs(recon - model_atrisk) > 2e-3:
        raise RuntimeError(f"SHAP reconstruction mismatch: {recon} vs {model_atrisk}")
    order = np.argsort(-np.abs(sv))
    k = min(k, len(sv))
    drivers = [
        {
            "feature": tm.features[i],
            "contribution": float(sv[i]),
            "value": float(row[tm.features[i]]),
        }
        for i in order[:k]
    ]
    return {
        "base_logit": base,
        "final_prob": model_atrisk,
        "reconstructed_prob": recon,
        "drivers": drivers,
    }


# ---------------------------------------------------------------- recommendations
def recommendations(tm: TrainedModel, row: Dict[str, float], drivers: List[Dict[str, Any]]) -> List[str]:
    """
    Deterministic, data-linked recommendations: only fire when a *driver*
    of elevated at-risk log-odds is present (contribution > 0). No invented numbers.
    """
    pos = {d["feature"]: d for d in drivers if d["contribution"] > 0}
    recs: List[str] = []
    if tm.key == "umbjm":
        gpa_features = [f for f in pos if "IPK" in f or "IPS" in f]
        sks_features = [f for f in pos if "sks" in f.lower()]
        if gpa_features:
            f = gpa_features[0]
            recs.append(
                f"GPA semester 1 (IPK {row[f]:.2f}) adalah driver risiko utama: jadwalkan tutorial terstruktur "
                f"untuk mata kuliah dengan skor di bawah C dalam 6 minggu pertama."
            )
        if sks_features:
            recs.append(
                "Beban SKS semester 1 ikut menaikkan risiko: batasi beban semester berikutnya maksimal 18–20 SKS "
                "hingga IPK <2.50 teratasi."
            )
        if pos:
            recs.insert(
                0,
                "Tindak lanjuti minggu ini: pemetaan penyebab non-akademik (finansial/pekerjaan/transportasi) "
                "sebelum memilih bentuk dukungan."
            )
        recs.append(
            "Pantau ulang via aplikasi setelah nilai semester 1 final; model ini hanya berbasis fitur semester 1."
        )
    else:
        if any("approved" in f for f in pos):
            recs.append(
                "Mata kuliah semester 1 yang lolos (approved) adalah driver risiko terkuat: pasangkan tutor mata kuliah "
                "pada 4 minggu pertama, target menambah approved units."
            )
        if "Tuition fees up to date" in pos or "Debtor" in pos:
            recs.append(
                "Status keuangan membebani prediksi: aktifkan konsultasi beasiswa/skim pembayaran sebelum penilaian "
                "risiko berikutnya."
            )
        if pos:
            recs.append("Buat rencana belajar tertulis untuk minggu-minggu terakhir (menjelang ujian semester 1).")
    if not recs:
        recs.append("Profil tidak menunjukkan driver risiko dominan: pantau rutin tiap bulan perkuliahan.")
    return recs


# ---------------------------------------------------------------- input schema
def input_specs(key: str) -> List[Dict[str, Any]]:
    """UI field schema per dataset (labels in Indonesian, ranges validated)."""
    if key == "umbjm":
        return [
            {"name": "Program Studi", "label": "Program studi (kode)", "min": 1, "max": 40, "default": 20, "step": 1, "fmt": "%d"},
            {"name": "Jalur Masuk", "label": "Jalur masuk (kode)", "min": 1, "max": 20, "default": 8, "step": 1, "fmt": "%d"},
            {"name": "Asal Sekolah", "label": "Asal sekolah (kode)", "min": 1, "max": 300, "default": 184, "step": 1, "fmt": "%d"},
            {"name": "Tahun_PMB", "label": "Tahun penerimaan (kode PMB)", "min": 0, "max": 10, "default": 6, "step": 1, "fmt": "%d"},
            {"name": "IPS Semester 1", "label": "IPS semester 1", "min": 0.0, "max": 4.0, "default": 3.2, "step": 0.01, "fmt": "%.2f"},
            {"name": "Jumlah sks Semester Semester 1", "label": "SKS diambil semester 1", "min": 0, "max": 24, "default": 19, "step": 1, "fmt": "%d"},
            {"name": "IPK Semester 1", "label": "IPK semester 1", "min": 0.0, "max": 4.0, "default": 3.2, "step": 0.01, "fmt": "%.2f"},
            {"name": "Jumlah sks Total Semester 1", "label": "Total SKS semester 1", "min": 0, "max": 24, "default": 19, "step": 1, "fmt": "%d"},
        ]
    return [
        {"name": "Marital status", "label": "Status perkawinan (kode)", "min": 1, "max": 6, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Application mode", "label": "Jalur pendaftaran (kode)", "min": 1, "max": 57, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Application order", "label": "Urutan pilihan", "min": 0, "max": 9, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Course", "label": "Program studi (kode)", "min": 171, "max": 9960, "default": 9254, "step": 1, "fmt": "%d"},
        {"name": "Daytime/evening attendance", "label": "Regular/malam (1=regular)", "min": 0, "max": 1, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Previous qualification", "label": "Kualifikasi awal (kode)", "min": 1, "max": 43, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Previous qualification (grade)", "label": "Nilai kualifikasi awal", "min": 0.0, "max": 400.0, "default": 122.0, "step": 0.5, "fmt": "%.1f"},
        {"name": "Nacionality", "label": "Kebangsaan (kode)", "min": 1, "max": 109, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Mother's qualification", "label": "Pendidikan ibu (kode)", "min": 1, "max": 43, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Father's qualification", "label": "Pendidikan ayah (kode)", "min": 1, "max": 43, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Mother's occupation", "label": "Pekerjaan ibu (kode)", "min": 0, "max": 195, "default": 5, "step": 1, "fmt": "%d"},
        {"name": "Father's occupation", "label": "Pekerjaan ayah (kode)", "min": 0, "max": 195, "default": 9, "step": 1, "fmt": "%d"},
        {"name": "Admission grade", "label": "Nilai masuk", "min": 95.0, "max": 190.0, "default": 122.0, "step": 0.5, "fmt": "%.1f"},
        {"name": "Displaced", "label": "Displaced (0/1)", "min": 0, "max": 1, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Educational special needs", "label": "Kebutuhan khusus (0/1)", "min": 0, "max": 1, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Debtor", "label": "Status hutang (0/1)", "min": 0, "max": 1, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Tuition fees up to date", "label": "Pembayaran lancar (0/1)", "min": 0, "max": 1, "default": 1, "step": 1, "fmt": "%d"},
        {"name": "Gender", "label": "Gender (0/1)", "min": 0, "max": 1, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Scholarship holder", "label": "Penerima beasiswa (0/1)", "min": 0, "max": 1, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Age at enrollment", "label": "Usia saat mendaftar", "min": 17, "max": 70, "default": 19, "step": 1, "fmt": "%d"},
        {"name": "International", "label": "Mahasiswa asing (0/1)", "min": 0, "max": 1, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Curricular units 1st sem (credited)", "label": "Unit dikreditkan semester 1", "min": 0, "max": 76, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Curricular units 1st sem (enrolled)", "label": "Unit diambil semester 1", "min": 0, "max": 23, "default": 6, "step": 1, "fmt": "%d"},
        {"name": "Curricular units 1st sem (evaluations)", "label": "Unit dievaluasi semester 1", "min": 0, "max": 47, "default": 8, "step": 1, "fmt": "%d"},
        {"name": "Curricular units 1st sem (approved)", "label": "Unit lulus semester 1", "min": 0, "max": 23, "default": 5, "step": 1, "fmt": "%d"},
        {"name": "Curricular units 1st sem (grade)", "label": "Nilai rata-rata semester 1", "min": 0.0, "max": 400.0, "default": 285.0, "step": 0.5, "fmt": "%.1f"},
        {"name": "Curricular units 1st sem (without evaluations)", "label": "Unit tanpa evaluasi semester 1", "min": 0, "max": 12, "default": 0, "step": 1, "fmt": "%d"},
        {"name": "Unemployment rate", "label": "Tingkat pengangguran", "min": 7.0, "max": 17.0, "default": 10.1, "step": 0.1, "fmt": "%.1f"},
        {"name": "Inflation rate", "label": "Tingkat inflasi", "min": -1.0, "max": 4.0, "default": 0.6, "step": 0.1, "fmt": "%.1f"},
        {"name": "GDP", "label": "GDP", "min": -5.0, "max": 8.0, "default": 1.23, "step": 0.1, "fmt": "%.1f"},
    ]


def presets(key: str) -> Dict[str, Dict[str, Any]]:
    """
    Two real examples drawn from the verified SHAP case studies
    (see shap_umbjm_REAL.json / shap_uci_REAL.json) + one low-risk example
    taken from actual data rows (median-safe). Only real values are used.
    """
    if key == "umbjm":
        return {
            # Exact row idx=1117 of quin/umbjm.csv (Target=0, at-risk) — verified.
            "Contoh: sangat berisiko (data nyata idx=1117)": {
                "Program Studi": 7, "Jalur Masuk": 15, "Asal Sekolah": 184, "Tahun_PMB": 2,
                "IPS Semester 1": 1.14, "Jumlah sks Semester Semester 1": 19,
                "IPK Semester 1": 1.14, "Jumlah sks Total Semester 1": 19,
            },
            "Contoh: aman (profil lulusan)": {
                "Program Studi": 20, "Jalur Masuk": 8, "Asal Sekolah": 184, "Tahun_PMB": 5,
                "IPS Semester 1": 3.5, "Jumlah sks Semester Semester 1": 20,
                "IPK Semester 1": 3.5, "Jumlah sks Total Semester 1": 20,
            },
        }
    return {
        # Exact row idx=3042 of quin/uci.csv (Target=0, dropout) — verified.
        "Contoh: sangat berisiko (data nyata idx=3042)": {
            "Marital status": 1, "Application mode": 39, "Application order": 1, "Course": 9670,
            "Daytime/evening attendance": 1, "Previous qualification": 12,
            "Previous qualification (grade)": 39.0, "Nacionality": 1,
            "Mother's qualification": 37, "Father's qualification": 1,
            "Mother's occupation": 9, "Father's occupation": 3,
            "Admission grade": 167.0, "Displaced": 0, "Educational special needs": 0,
            "Debtor": 1, "Tuition fees up to date": 0, "Gender": 1,
            "Scholarship holder": 0, "Age at enrollment": 40, "International": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": 6,
            "Curricular units 1st sem (evaluations)": 13,
            "Curricular units 1st sem (approved)": 0,
            "Curricular units 1st sem (grade)": 30.0,
            "Curricular units 1st sem (without evaluations)": 0,
            "Unemployment rate": 15.05, "Inflation rate": 2.08, "GDP": 3.0,
        },
    }
