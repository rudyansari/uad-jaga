# -*- coding: utf-8 -*-
"""
UAD-JAGA — Sistem pendamping deteksi dini risiko ketidaklulusan mahasiswa.
Semua angka: model terlatih nyata + metrik terverifikasi dari Riset/reanalysis/.
Tidak ada angka simulasi.

Run:  streamlit run app_uad_jaga.py
"""
from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from uad_jaga_core import (
    ArtifactNotFound,
    DATASETS,
    exact_shap_contributions,
    input_specs,
    load_artifacts,
    predict,
    presets,
    recommendations,
    verified_metrics,
)

# --------------------------------------------------------------------- page
st.set_page_config(
    page_title="UAD-JAGA — Deteksi Dini Risiko Kelulusan",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- theme
st.markdown(
    """
<style>
:root{
  --uj-primary:#1B4F8A; --uj-primary-600:#16406F;
  --uj-bg:#FFFFFF; --uj-surface:#F5F7FA; --uj-border:#D9E2EC;
  --uj-text:#1F2933; --uj-muted:#52606D;
  --uj-low:#2E7D32; --uj-med:#ED6C02; --uj-high:#C62828;
}
html, body, [class*="css"]{color:var(--uj-text);}
.uj-header{padding:1.1rem 1.4rem;background:var(--uj-primary);color:#fff;border-radius:10px;margin-bottom:1rem;}
.uj-header h1{color:#fff;font-size:1.5rem;margin:0;}
.uj-header p{color:#DCE9F7;margin:.25rem 0 0 0;font-size:.95rem;}
.uj-card{background:var(--uj-surface);border:1px solid var(--uj-border);border-radius:10px;padding:1rem 1.2rem;}
.uj-tier-low{color:var(--uj-low);font-weight:700;}
.uj-tier-medium{color:var(--uj-med);font-weight:700;}
.uj-tier-high{color:var(--uj-high);font-weight:700;}
.uj-metric{border:1px solid var(--uj-border);border-radius:8px;padding:.75rem 0;text-align:center;background:#fff;}
.uj-metric b{font-size:1.25rem;}
.uj-note{font-size:.85rem;color:var(--uj-muted);}
.uj-rec li{margin-bottom:.45rem;line-height:1.45;}
.uj-divider{border:none;border-top:1px solid var(--uj-border);margin:1.25rem 0;}
section[data-testid="stSidebar"]>div{background:#FBFCFE;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="uj-header"><h1>🎓 UAD-JAGA</h1>
<p><b>J</b>ujur <b>A</b>dvisory for <b>G</b>raduation-risk <b>A</b>nalysis — deteksi dini risiko ketidaklulusan
berbasis fitur semester&nbsp;1, dengan audit kebocoran label. Alat pendukung keputusan — bukan pengganti penilai manusia.</p></div>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------- state
if "result" not in st.session_state:
    st.session_state.result = None
if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = "umbjm"

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("⚙️ Konfigurasi")
    dataset_label = st.selectbox(
        "Konteks dataset",
        options=list(DATASETS.keys()),
        format_func=lambda k: DATASETS[k]["label"],
        index=0 if st.session_state.dataset_key == "umbjm" else 1,
        help="Model berbeda per dataset; keduanya dilatih HANYA dari fitur semester 1.",
    )
    if dataset_label != st.session_state.dataset_key:
        st.session_state.dataset_key = dataset_label
        st.session_state.result = None

    try:
        tm = load_artifacts(dataset_label)
    except ArtifactNotFound as e:
        st.error(str(e))
        st.stop()

    spec = tm.spec
    st.markdown(
        f"""<div class="uj-card" style="margin-top:.75rem">
<b>Kartu model</b><br>
<span class="uj-note">Baris tugas: {spec['n_rows']:,} · fitur: {len(spec['features'])} ·
at-risk: {spec['at_risk_rate']*100:.1f}%<br>Label: 0=at-risk · 1=safe</span></div>""",
        unsafe_allow_html=True,
    )

    st.subheader("📌 Preset")
    ps = presets(dataset_label)
    choice = st.selectbox("Isi contoh nyata", ["— pilih —"] + list(ps.keys()))
    if st.button("Terapkan preset", width="stretch") and choice != "— pilih —":
        for k, v in ps[choice].items():
            st.session_state[f"in_{k}"] = v
        st.session_state.result = None
        st.rerun()

    b0, b1 = DATASETS[dataset_label]["bands"]
    st.markdown("---")
    st.subheader("🧭 Ambang operasional")
    st.markdown(
        f"""<div class="uj-note">
Ambang tier: LOW&nbsp;&lt;&nbsp;{b0:.2f} &nbsp;·&nbsp; MEDIUM&nbsp;{b0:.2f}–{b1:.2f} &nbsp;·&nbsp; HIGH&nbsp;≥&nbsp;{b1:.2f}.<br>
Ambang ini <b>operasional</b> (dipilih, bukan dituning data); titik-titik kerja yang
diturunkan data (mis. TPR@FPR≤5%) tercatat di tab <i>Metrik Terverifikasi</i>.</div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------- helpers
def _risk_gauge_png(risk: float, tier: str) -> "str | None":
    """Render a clean horizontal gauge; saved to /tmp and returned path."""
    fig, ax = plt.subplots(figsize=(8, 0.9), dpi=160)
    for a, b, c in [(0, 0.3, "#A5D6A7"), (0.3, 0.7, "#FFE0B2"), (0.7, 1.0, "#EF9A9A")]:
        ax.barh([0], [b - a], left=a, height=0.6, color=c)
    ax.axvline(
        risk,
        color={"LOW": "#2E7D32", "MEDIUM": "#ED6C02", "HIGH": "#C62828"}[tier],
        lw=3,
    )
    ax.text(risk, 0.5, f"{risk:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for t, x in [("LOW", 0.15), ("MEDIUM", 0.5), ("HIGH", 0.85)]:
        ax.text(x, -0.55, t, ha="center", fontsize=8, color="#52606D")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.axis("off")
    path = f"/tmp/uj_gauge_{dataset_label}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _shap_png(shap_out, dataset_label: str) -> str:
    drivers = shap_out["drivers"]
    top = drivers[:6][::-1]
    labels = [f"{d['feature']} = {d['value']:.4g}" for d in top]
    vals = [d["contribution"] for d in top]
    starts = np.cumsum([shap_out["base_logit"]] + vals[:-1]).tolist()
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    colors = ["#C62828" if s > 0 else "#2E7D32" for s in vals]
    ax.barh(labels, vals, left=starts, color=colors, alpha=0.88, height=0.62)
    ax.axvline(0, color="#52606D", ls=":", lw=1)
    ax.set_title(
        f"Dorongan risiko (Tree SHAP exact) — base {shap_out['base_logit']:+.2f} → "
        f"margin {shap_out['base_logit']+sum(vals):+.2f} → p={shap_out['final_prob']:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Kontribusi ke log-odds at-risk (positif = menaikkan risiko)")
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    path = f"/tmp/uj_shap_{dataset_label}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------- tabs
tab_in, tab_view, tab_shap, tab_rec, tab_metrics, tab_about = st.tabs(
    ["📝 Input & Prediksi", "📊 Hasil", "🧠 Penjelasan", "💡 Rekomendasi", "✅ Metrik Terverifikasi", "ℹ️ Tentang"]
)

with tab_in:
    st.markdown("#### Data mahasiswa (fitur semester 1 saja)")
    st.caption(
        "Model sengaja TIDAK membaca fitur semester lanjut (mis. IPS sem-2) — "
        "termasuknya fitur itu menaikkan AUC ke ≈1.000 pada data internal kami lewat kebocoran label "
        "(lihat tab Metrik Terverifikasi)."
    )
    fields = input_specs(dataset_label)
    cols = st.columns(3 if len(fields) <= 12 else 4)
    row = {}
    for i, fdef in enumerate(fields):
        key = f"in_{fdef['name']}"
        default = st.session_state.get(key, fdef["default"])
        with cols[i % len(cols)]:
            if fdef["step"] == 1:
                v = st.number_input(
                    fdef["label"],
                    min_value=int(fdef["min"]),
                    max_value=int(fdef["max"]),
                    value=int(default),
                    step=1,
                    key=f"in_{fdef['name']}",
                )
            else:
                v = st.number_input(
                    fdef["label"],
                    min_value=float(fdef["min"]),
                    max_value=float(fdef["max"]),
                    value=float(default),
                    step=float(fdef["step"]),
                    format=fdef["fmt"],
                    key=f"in_{fdef['name']}",
                )
        row[fdef["name"]] = float(v)

    c1, c2 = st.columns([1, 3])
    run = c1.button("🎯 Analisis risiko", type="primary", width="stretch")
    if run:
        res = predict(tm, row)
        try:
            sh = exact_shap_contributions(tm, row)
            rec = recommendations(tm, row, sh["drivers"])
        except Exception as e:  # pragma: no cover - defensive UI state
            sh, rec = None, [f"(SHAP tidak tersedia: {e})"]
        st.session_state.result = {
            "input": row,
            "predict": res,
            "shap": sh,
            "recs": rec,
            "dataset": dataset_label,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

with tab_view:
    if st.session_state.result is None:
        st.info("Belum ada hasil — isi tab **Input & Prediksi** lalu klik **Analisis risiko**.")
    else:
        R = st.session_state.result
        risk, tier = R["predict"]["risk"], R["predict"]["tier"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="uj-metric"><span class="uj-note">Risiko at-risk</span><br>'
                f'<b class="uj-tier-{tier.lower()}">{risk:.1%}</b></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="uj-metric"><span class="uj-note">Kategori</span><br>'
                f'<b class="uj-tier-{tier.lower()}">{tier}</b></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="uj-metric"><span class="uj-note">Probabilitas safe</span><br>'
                f'<b>{R["predict"]["p_safe"]:.1%}</b></div>',
                unsafe_allow_html=True,
            )
        st.image(_risk_gauge_png(risk, tier), width="stretch")
        st.caption(
            f"Model: {DATASETS[R['dataset']]['label']} · {R['predict']['features_used']} fitur · "
            f"dianalisis {R['time']}"
        )

        st.markdown("<hr class='uj-divider'>", unsafe_allow_html=True)
        st.markdown("**Ekspor hasil penilaian**")
        out = {
            "timestamp": R["time"],
            "dataset": R["dataset"],
            "risk": risk,
            "tier": tier,
            **{f"x_{k}": v for k, v in R["input"].items()},
        }
        rec_csv = pd.DataFrame([out]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Unduh CSV",
            rec_csv,
            file_name=f"uad_jaga_assessment_{R['time'][:10]}_{R['dataset']}.csv",
            mime="text/csv",
        )

with tab_shap:
    if st.session_state.result is None or st.session_state.result["shap"] is None:
        st.info("Belum ada penjelasan — jalankan analisis di tab Input terlebih dahulu.")
    else:
        R = st.session_state.result
        sh = R["shap"]
        st.image(_shap_png(sh, R["dataset"]), width="content")
        df_drivers = pd.DataFrame(sh["drivers"])
        df_drivers["arah"] = df_drivers["contribution"].map(
            lambda s: "⬆ menaikkan risiko" if s > 0 else "⬇ menurunkan risiko"
        )
        st.dataframe(
            df_drivers[["feature", "value", "contribution", "arah"]].rename(
                columns={"feature": "Fitur", "value": "Nilai", "contribution": "Kontribusi", "arah": "Arah"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Verifikasi rekonstruksi: p(model)={sh['final_prob']:.3f} ≈ "
            f"p(SHAP)={sh['reconstructed_prob']:.3f} — Tree SHAP exact."
        )

with tab_rec:
    if st.session_state.result is None:
        st.info("Belum ada rekomendasi — jalankan analisis di tab Input terlebih dahulu.")
    else:
        R = st.session_state.result
        st.markdown(
            "Rekomendasi hanya dipicu oleh driver SHAP dengan kontribusi **positif** — "
            "tidak ada klaim otomatis yang tidak bersumber dari penjelasan terukur."
        )
        recs = R["recs"]
        st.markdown("<ul class='uj-rec'>" + "".join(f"<li>{x}</li>" for x in recs) + "</ul>", unsafe_allow_html=True)
        st.caption(
            "Catatan: bandingkan saran dengan penilaian profesional Anda. Sistem mengandung bias historis "
            "sepanjang fitur semester 1 dan target biner; gunakan sebagai pendukung, bukan penilai tunggal."
        )

with tab_metrics:
    st.markdown("#### Metrik terverifikasi (dari perangkat ulang CV 5-fold, pooled out-of-fold, seed 42)")
    vm = verified_metrics()
    if "leakage_audit_REAL.csv" in vm["files"]:
        st.markdown("**· Audit kebocoran (measurement-forward leakage)**")
        st.dataframe(pd.DataFrame(vm["files"]["leakage_audit_REAL.csv"]), width="stretch", hide_index=True)
        st.caption(
            "EARLY-WARNING: model yang hanya diberi fitur semester 1. FULL: model dengan fitur semester lanjut "
            "(ditambahkan di sini hanya untuk audit — bukan dipakai untuk prediksi)."
        )
    if "cv_benchmark_REAL.json" in vm["files"]:
        st.markdown("**· Benchmark model (pooled OOF)**")
        rows = []
        b = vm["files"]["cv_benchmark_REAL.json"]
        for dn in b:
            for gname in b[dn]:
                if "Sem1" not in gname:
                    continue
                for mname, mv in b[dn][gname]["models"].items():
                    p = mv["pooled"]
                    fs = mv["fold_mean_std"]
                    rows.append({
                        "dataset": dn,
                        "model": mname,
                        "AUC(pooled)": round(p["auc"], 3),
                        "AUC fold mean±std (ddof=1)": f"{fs['auc'][0]:.3f}±{fs['auc'][1]:.3f}",
                        "P": round(p["precision"], 3),
                        "R": round(p["recall"], 3),
                        "F1": round(p["f1"], 3),
                        "Acc": round(p["accuracy"], 3),
                        "P@100": round(mv["precision_at_100"], 2) if mv["precision_at_100"] is not None else "n/a",
                    })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if "calibration_REAL.json" in vm["files"]:
        st.markdown("**· Kalibrasi (holdout 15%)**")
        st.dataframe(pd.DataFrame(vm["files"]["calibration_REAL.json"]).T.round(4), width="stretch")
    if "human_eval_descriptive_REAL.csv" in vm["files"]:
        st.markdown("**· Evaluasi manusia (n=48 responden, 2.160 rating)**")
        st.dataframe(pd.DataFrame(vm["files"]["human_eval_descriptive_REAL.csv"]), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(vm["files"]["human_eval_effects_REAL.csv"]), width="stretch", hide_index=True)

with tab_about:
    st.markdown(
        """#### Tentang UAD-JAGA
**UAD-JAGA** (*JuJur Advisory for Graduation-risk Analysis*; *jaga*, bahasa Indonesia: *mengawasi*) adalah
alat pendukung keputusan untuk penasihat akademik. Prinsipnya tercermin di aplikasi ini:

- **Fitur penuh kejujuran** — hanya fitur semester 1 yang dipakai; fitur semester lanjut sengaja dibuang setelah
  audit menunjukkan adanya measurement-forward leakage.
- **Angka terverifikasi** — setiap metrik pada tab *Metrik Terverifikasi* berasal dari proses ulang pada data asli
  (5-fold stratified CV); tidak ada angka simulasi atau hardcoded.
- **Penjelasan yang dapat direproduksi** — kontribusi harus dapat direkonstruksi ulang ke probabilitas model
  (Tree SHAP exact); jika tidak cocok, aplikasi menolak memberi penjelasan.
- **Keterbatasan dinyatakan terbuka** — model tidak menangkap faktor tidak tercatat (motivasi, kesehatan mental,
  kondisi keluarga), dan target kolektifnya adalah data historis.

#### Standar operasional dalam UI
- Skor risiko = probabilitas model untuk kelas *at-risk* (label 0).
- Ambang tier operasional (bukan dituning): LOW < 0.30 ≤ MEDIUM < 0.70 ≤ HIGH.
- Rekomendasi memicu hanya dari driver SHAP positif — tidak ada klaim otomatis yang tidak bersumber.

**Tim**: Rudy Ansari, Sunardi, Imam Riadi — Universitas Ahmad Dahlan & Universitas Muhammadiyah Banjarmasin."""
    )

st.markdown("<hr class='uj-divider'>", unsafe_allow_html=True)
st.caption("UAD-JAGA · deteksi dini risiko kelulusan · hanya untuk tujuan pendukung keputusan · " + datetime.now().strftime("%Y"))
