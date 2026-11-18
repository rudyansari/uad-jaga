# -*- coding: utf-8 -*-
"""
UAD-JAGA — deteksi dini risiko gagal lulus mahasiswa.
Angka & model: terverifikasi (Riset/reanalysis). Tidak ada angka simulasi.

Run: streamlit run app_uad_jaga.py
"""
from __future__ import annotations

import base64
import os
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

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "assets", "uj_logo.png")

# ------------------------------------------------------------------ page + theme
st.set_page_config(
    page_title="UAD-JAGA — Deteksi dini risiko kelulusan",
    page_icon=LOGO if os.path.exists(LOGO) else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

:root{
  --ink:#122B45;  --ink-soft:#3A4C63;  --paper:#F6F3ED;  --card:#FFFFFF;
  --border:#E6DFD1;  --accent:#C87D3B;
  --low:#3E7A55;  --low-bg:#E6F0E8;  --med:#A9761F;  --med-bg:#F7EBD2;
  --high:#A94335;  --high-bg:#F6E0DC;
  --serif:'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans:'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
html, body { background: var(--paper); }
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: var(--sans); color: var(--ink); }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{ background: var(--paper) !important; }

/* Sembunyikan chrome bawaan framework (Deploy, kebab, header, footer) */
[data-testid="stHeader"], #MainMenu, .stDeployButton,
[data-testid="stToolbarActions"], [data-testid="stFooter"], footer{ display:none !important; height:0 !important; }

/* Header band */
.uj-topbar{
  display:flex; align-items:center; gap:14px;
  padding: 18px 22px; background: linear-gradient(135deg,#122B45 0%, #1B3A5B 100%);
  border-radius: 14px; margin-bottom: 10px; color: #F6F3ED;
  box-shadow: 0 6px 18px rgba(18,43,69,.14);
}
.uj-topbar img{ width:52px; height:52px; border-radius:12px; flex-shrink:0;}
.uj-topbar > div{ min-width:0; flex:1 1 auto; }
.uj-topbar > div p{ max-width:100%; overflow-wrap:anywhere;}
.uj-topbar h1{ font-family:var(--serif); font-size:1.62rem; margin:0; color:#F6F3ED; letter-spacing:.2px;}
.uj-topbar p{ margin:2px 0 0 0; color:#C6D2E0; font-size:.83rem; max-width:640px; line-height:1.45;}
.uj-badge{
  margin-left:auto; background:rgba(246,243,237,.12); border:1px solid rgba(246,243,237,.25);
  padding:6px 12px; border-radius:999px; font-size:.72rem; color:#EDE6D9;
  flex:0 0 auto; white-space:normal; line-height:1.3; text-align:center; min-width:110px;
}

/* Cards */
.uj-card{
  background: var(--card); border:1px solid var(--border); border-radius:12px;
  padding:16px 18px; box-shadow: 0 1px 2px rgba(18,43,69,.05);
  margin-bottom:12px;
}
.uj-card h4{ font-family:var(--serif); margin:0 0 8px 0; font-size:1.02rem; color: var(--ink); }
.uj-muted{ color: var(--ink-soft); font-size:.82rem; line-height:1.5; }

/* Big risk display */
.uj-risk-num{ font-family:var(--serif); font-size:3.2rem; font-weight:700; line-height:1.1; }
.uj-tierchip{ display:inline-block; padding:4px 14px; border-radius:999px; font-weight:600; font-size:.85rem; }
.tier-LOW    { background:var(--low-bg);  color:var(--low);   border:1px solid #BFDCC5;}
.tier-MEDIUM { background:var(--med-bg);  color:var(--med);   border:1px solid #EAD7AC;}
.tier-HIGH   { background:var(--high-bg); color:var(--high);  border:1px solid #EBC2BA;}
.uj-line-human{ font-size:1.02rem; color:var(--ink); margin-top:6px;}

/* Legend chips */
.uj-chips{ display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;}
.uj-chip{ font-size:.72rem; padding:3px 10px; border-radius:999px; border:1px solid var(--border); background:#FBFAF6; color:var(--ink-soft);}

/* Tabs: pill style */
.stTabs [data-baseweb="tab-list"]{ gap:4px; background:#EFEBE1; padding:5px; border-radius:12px; }
.stTabs [data-baseweb="tab"]{
  border-radius:9px; background:transparent; color:var(--ink-soft);
  font-weight:500; padding:8px 14px; height:auto; border:none;
}
.stTabs [data-baseweb="tab"]:hover{ background:#FFFFFFAA; }
.stTabs [aria-selected="true"]{
  background:#FFFFFF !important; color:var(--ink) !important;
  box-shadow: 0 1px 3px rgba(18,43,69,.12); font-weight:600;
}

/* Buttons */
button[kind="primary"]{
  background: var(--ink) !important; border-radius:10px !important;
  border:1px solid var(--ink) !important; font-weight:600 !important;
}
button[kind="primary"]:hover{ background:#1B3A5B !important; }

/* Sidebar */
section[data-testid="stSidebar"]{ background:#F0EBDF; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4{ font-family:var(--serif); color:var(--ink); }
section[data-testid="stSidebar"] label{ color: var(--ink) !important; opacity:1 !important; font-weight:500; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{ color: var(--ink) !important; opacity:1 !important; }

/* Metric tables */
.stDataFrame{ border:1px solid var(--border); border-radius:10px; }

/* Small print */
.uj-smallprint{ font-size:.74rem; color:#7A8798; line-height:1.5; }
.uj-footer{ margin-top:36px; padding-top:14px; border-top:1px solid var(--border);
  font-size:.76rem; color:#8A94A3; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;}
.uj-rec li{ margin-bottom:.55rem; line-height:1.55; }
.uj-empty{ padding:38px 22px; text-align:center; color:var(--ink-soft); }
</style>
""",
    unsafe_allow_html=True,
)

if os.path.exists(LOGO):
    _logo_b64 = base64.b64encode(open(LOGO, "rb").read()).decode()
    _logo_html = f'<img src="data:image/png;base64,{_logo_b64}"/>'
else:
    _logo_html = ""

st.markdown(
    f"""
<div class="uj-topbar">
  {_logo_html}
  <div>
    <h1>UAD-JAGA</h1>
    <p>Deteksi dini risiko gagal lulus — dibaca murni dari data semester pertama,
    tanpa melirik semester berikut. Alat bantu keputusan, bukan hakim akhir.</p>
  </div>
  <div class="uj-badge">model terverifikasi · v1.0</div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ state
if "result" not in st.session_state:
    st.session_state.result = None
if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = "umbjm"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Setelan")
    dataset_label = st.selectbox(
        "Konteks data",
        options=list(DATASETS.keys()),
        format_func=lambda k: DATASETS[k]["label"],
        index=0 if st.session_state.dataset_key == "umbjm" else 1,
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
        f"""<div class="uj-card">
<h4>Kartu model</h4>
<span class="uj-muted">{DATASETS[dataset_label]['desc']}<br>
{spec['n_rows']:,} mahasiswa · {len(spec['features'])} fitur · {spec['at_risk_rate']*100:.1f}% at-risk<br>
label <b>0</b> = at-risk · label <b>1</b> = selesai</span></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Contoh nyata")
    ps = presets(dataset_label)
    choice = st.selectbox("Isi otomatis dari kasus asli", ["— pilih sendiri —"] + list(ps.keys()))
    if st.button("Pakai contoh ini", width="stretch") and choice != "— pilih sendiri —":
        for k, v in ps[choice].items():
            st.session_state[f"in_{k}"] = v
        st.session_state.result = None
        st.rerun()

    b0, b1 = DATASETS[dataset_label]["bands"]
    st.markdown("### Ambang tier")
    st.markdown(
        f"""<div class="uj-card">
<span class="uj-muted">Tier ditetapkan operasional (bukan dituning hasil):
LOW&nbsp;&lt;&nbsp;{b0:.2f} ≤ MEDIUM&nbsp;&lt;&nbsp;{b1:.2f} ≤ HIGH. Titik kerja yang diturunkan data
(seperti batas alarm 5%) ada di tab <i>Angka &amp; bukti</i>.</span>
<div class="uj-chips">
<span class="uj-chip">LOW · hijau</span><span class="uj-chip">MEDIUM · kuning</span><span class="uj-chip">HIGH · merah</span>
</div></div>""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ plot helpers
INK = "#122B45"


def _donut_png(risk: float, tier: str, key: str) -> str:
    col = {"LOW": "#3E7A55", "MEDIUM": "#A9761F", "HIGH": "#A94335"}[tier]
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=180)
    fig.patch.set_facecolor("none")
    ax.pie(
        [risk, 1 - risk],
        startangle=90,
        counterclock=False,
        colors=[col, "#EFEAE0"],
        wedgeprops=dict(width=0.34, edgecolor="none"),
    )
    ax.set(aspect="equal", xticks=[], yticks=[], frame_on=False)
    path = f"/tmp/uj_donut_{key}.png"
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def _shap_png(shap_out, key: str) -> str:
    drivers = shap_out["drivers"][:6][::-1]
    labels = [f"{d['feature']} = {d['value']:.4g}" for d in drivers]
    vals = [d["contribution"] for d in drivers]
    starts = np.cumsum([shap_out["base_logit"]] + vals[:-1]).tolist()
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=180)
    fig.patch.set_facecolor("none")
    colors = ["#A94335" if s > 0 else "#3E7A55" for s in vals]
    ax.barh(labels, vals, left=starts, color=colors, alpha=0.92, height=0.6)
    ax.axvline(0, color="#9AA5B1", ls=":", lw=1)
    ax.set_title(
        f"base {shap_out['base_logit']:+.2f} → margin {shap_out['base_logit']+sum(vals):+.2f} → p = {shap_out['final_prob']:.3f}",
        fontsize=11, fontweight="bold", color=INK, loc="left", pad=12,
    )
    ax.set_xlabel("tarikan ke arah at-risk (log-odds)", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    path = f"/tmp/uj_shap_{key}.png"
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return path


def _tier_sentence(tier: str) -> str:
    return {
        "HIGH": "Risiko tinggi — jangan ditunda. Mulai dari penyebab akademik terbesar, lalu cek beban SKS dan dukungan non-akademik.",
        "MEDIUM": "Belum aman tapi masih bisa dijaga — pasang pantauan rutin mingguan.",
        "LOW": "Kondisi terkendali — pantauan ringan tiap bulan cukup.",
    }[tier]


# ------------------------------------------------------------------ tabs
tab_in, tab_view, tab_shap, tab_rec, tab_metrics, tab_about = st.tabs(
    ["Isi data", "Skor saya", "Mengapa begini", "Langkah yang membantu", "Angka & bukti", "Tentang"]
)

with tab_in:
    st.markdown("## Isi data mahasiswa")
    st.markdown(
        """<span class="uj-muted">Cukup data semester pertama. Model <b>sengaja</b> tidak membaca
IPS semester dua — di kolom itulah angka sempurna AUC 1,000 sebenarnya mencuri pandang.</span>""",
        unsafe_allow_html=True,
    )
    st.write("")
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

    run = st.button("Hitung risiko", type="primary")
    if run:
        res = predict(tm, row)
        try:
            sh = exact_shap_contributions(tm, row)
            rec = recommendations(tm, row, sh["drivers"])
        except Exception as e:
            sh, rec = None, [f"(penjelasan tidak tersedia: {e})"]
        st.session_state.result = {
            "input": row,
            "predict": res,
            "shap": sh,
            "recs": rec,
            "dataset": dataset_label,
            "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        }

with tab_view:
    if st.session_state.result is None:
        st.markdown(
            """<div class="uj-card uj-empty">Belum ada hitungan.<br>
<span class="uj-muted">Isi dulu di tab «Isi data» — atau pakai contoh nyata di sidebar — lalu tekan «Hitung risiko».</span></div>""",
            unsafe_allow_html=True,
        )
    else:
        R = st.session_state.result
        risk, tier = R["predict"]["risk"], R["predict"]["tier"]
        c1, c2, c3 = st.columns([1.2, 1.4, 2.2])
        with c1:
            st.image(_donut_png(risk, tier, R["dataset"]), width=210)
        with c2:
            colr = "#3E7A55" if tier == "LOW" else ("#A9761F" if tier == "MEDIUM" else "#A94335")
            st.markdown(
                f"""<div class="uj-risk-num" style="color:{colr}">{risk*100:.1f}%</div>
<span class="uj-tierchip tier-{tier}">{tier}</span>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="uj-card" style="margin-top:26px">
<h4>Apa artinya</h4>
<div class="uj-line-human">{_tier_sentence(tier)}</div>
<div class="uj-smallprint" style="margin-top:8px">peluang at-risk {risk:.2%} · peluang selesai {R['predict']['p_safe']:.2%} ·
dihitung {R['time']} dengan {R['predict']['features_used']} fitur semester 1</div>
</div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        out = {
            "waktu": R["time"],
            "dataset": R["dataset"],
            "risiko": round(risk, 4),
            "tier": tier,
            **{f"x_{k}": v for k, v in R["input"].items()},
        }
        st.download_button(
            "Simpan hasil (CSV)",
            pd.DataFrame([out]).to_csv(index=False).encode("utf-8"),
            file_name=f"uad-jaga-{R['dataset']}-{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

with tab_shap:
    if st.session_state.result is None or st.session_state.result["shap"] is None:
        st.markdown(
            """<div class="uj-card uj-empty">Penjelasan belum dibuat.<br>
<span class="uj-muted">Jalankan perhitungan dulu — setiap kontribusi direkonstruksi balik ke probabilitas model; kalau tidak cocok, tidak ditampilkan.</span></div>""",
            unsafe_allow_html=True,
        )
    else:
        R = st.session_state.result
        sh = R["shap"]
        st.markdown("## Mengapa skor-nya begini")
        st.image(_shap_png(sh, R["dataset"]))
        df_drivers = pd.DataFrame(sh["drivers"])
        df_drivers["arah"] = df_drivers["contribution"].map(
            lambda s: "menaikkan risiko" if s > 0 else "menurunkan risiko"
        )
        st.dataframe(
            df_drivers[["feature", "value", "contribution", "arah"]].rename(
                columns={"feature": "fitur", "value": "nilai", "contribution": "kontribusi"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"cek rekonstruksi: p(model) {sh['final_prob']:.3f} ≈ p(SHAP) {sh['reconstructed_prob']:.3f} — Tree SHAP exact."
        )

with tab_rec:
    if st.session_state.result is None:
        st.markdown(
            """<div class="uj-card uj-empty">Belum ada saran.<br>
<span class="uj-muted">Saran hanya muncul dari penyebab risiko yang terukur di profil ini — bukan himbauan umum.</span></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## Langkah yang kemungkinan membantu")
        st.markdown(
            """<span class="uj-muted">Saran ini khusus dari profil yang diisi — hanya penyebab yang
benar-benar menarik risiko yang memicunya. Ukur ulang setelah nilai akhir semester 1 keluar.</span>""",
            unsafe_allow_html=True,
        )
        recs = st.session_state.result["recs"]
        st.markdown("<ul class='uj-rec'>" + "".join(f"<li>{x}</li>" for x in recs) + "</ul>", unsafe_allow_html=True)
        st.caption(
            "Gabungkan dengan pertimbangan Anda — model tidak melihat motivasi, kondisi keluarga, atau kesehatan."
        )

with tab_metrics:
    st.markdown("## Angka & bukti")
    st.markdown(
        """<span class="uj-muted">Angka ini <b>persis</b> sama dengan isi naskah paper —
dibaca langsung dari file artefaknya, bukan ketik tangan, bukan perkiraan.</span>""",
        unsafe_allow_html=True,
    )
    st.write("")
    vm = verified_metrics()
    if "leakage_audit_REAL.csv" in vm["files"]:
        st.markdown("#### Audit bocor data (measurement-forward leakage)")
        st.dataframe(pd.DataFrame(vm["files"]["leakage_audit_REAL.csv"]), width="stretch", hide_index=True)
        st.caption(
            "FULL = semua fitur (termasuk semester 2, hanya untuk audit). EARLY = fitur semester 1 saja — yang dipakai aplikasi ini."
        )
    if "cv_benchmark_REAL.json" in vm["files"]:
        st.markdown("#### Benchmark model (5-fold CV, pooled out-of-fold)")
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
                        "AUC": round(p["auc"], 3),
                        "fold AUC (mean±std)": f"{fs['auc'][0]:.3f}±{fs['auc'][1]:.3f}",
                        "P": round(p["precision"], 3),
                        "R": round(p["recall"], 3),
                        "F1": round(p["f1"], 3),
                        "Acc": round(p["accuracy"], 3),
                        "P@100": round(mv["precision_at_100"], 2),
                    })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if "calibration_REAL.json" in vm["files"]:
        st.markdown("#### Kalibrasi (holdout 15%)")
        st.dataframe(pd.DataFrame(vm["files"]["calibration_REAL.json"]).T.round(4), width="stretch")
    if "human_eval_descriptive_REAL.csv" in vm["files"]:
        st.markdown("#### Penilaian manusia (48 penasihat, 2.160 penilaian)")
        st.dataframe(pd.DataFrame(vm["files"]["human_eval_descriptive_REAL.csv"]), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(vm["files"]["human_eval_effects_REAL.csv"]), width="stretch", hide_index=True)

with tab_about:
    st.markdown("## Tentang UAD-JAGA")
    st.markdown(
        """**J**ujur **A**dvisory for **G**raduation-risk **A**nalysis — nama *jaga* dipinjam dari
bahasa Indonesia: *mengawasi*.

Aplikasi ini adalah sisi manusia dari riset kami. Prinsip yang kami pegang teguh:

- Hanya fitur semester pertama — bukan kebetulan, tapi kejujuran metodologis.
- Setiap angka yang tampil bisa direproduksi dari berkas di repositori terbuka (github.com/rudyansari/uad-jaga).
- Setiap penjelasan harus rekonsili dengan probabilitas model; kalau tidak cocok, tidak ditampilkan.
- Keterbatasan ditulis terbuka: model tidak mengenal motivasi, kondisi ekonomi, maupun kesehatan.

**Tim**: Rudy Ansari, Sunardi, Imam Riadi — Universitas Ahmad Dahlan & Universitas Muhammadiyah Banjarmasin.

Aplikasi ini adalah pendukung keputusan, bukan putusan.""",
    )

st.markdown(
    """<div class="uj-footer">
<span>UAD-JAGA · deteksi dini risiko kelulusan · angka terverifikasi, tanpa fabrikasi</span>
<span>Rudy Ansari · Sunardi · Imam Riadi · 2026</span>
</div>""",
    unsafe_allow_html=True,
)
