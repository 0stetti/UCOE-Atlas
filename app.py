"""
UCOE Atlas — Interactive platform for exploring UCOE candidates
in the human genome.

Developed by Elton Roger Ostetti (USP / Instituto Butantan)
Supervisors: Prof. Dr. Ana Maria Moro · Prof. Dr. Tânia Maria Manieri

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="UCOE Atlas",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — colors, CSS, Plotly template
# ══════════════════════════════════════════════════════════════════════════════
# ── Magazine palette (mirrors magazine_style.py) ─────────────────────────────
P_AQUA     = "#6FC3C8"   # primary data color
P_ROSE     = "#F29AA0"   # secondary
P_PEACH    = "#F6C47A"   # tertiary / accent
P_SKY      = "#88BFEB"   # quaternary
P_LAVENDER = "#C0A8DC"   # quinary
P_MINT     = "#8FCFA8"   # senary
P_INK      = "#1C1E26"   # near-black — titles
P_SLATE    = "#5D6470"   # mid-grey — body / labels
P_GHOST    = "#9CA3AF"   # light grey — captions
P_GRID     = "#E9ECEF"   # gridlines
P_RULE     = "#D1D5DB"   # borders / spines
P_BG       = "#FFFFFF"   # page background
P_BG_SUB   = "#F8F9FB"   # sidebar / card background
P_CREAM    = "#FFF8E8"   # warm cream callout

# Aliases kept for backward compat with existing page code
C_NAVY   = P_INK
C_BLUE   = P_AQUA
C_TEAL   = P_AQUA
C_GREEN  = P_MINT
C_AMBER  = P_PEACH
C_RED    = P_ROSE
C_PURPLE = P_LAVENDER
C_GRAY   = P_SLATE
C_BGCARD = P_BG
C_BORDER = P_RULE

PALETTE = [P_AQUA, P_ROSE, P_PEACH, P_SKY, P_LAVENDER, P_MINT,
           "#5BADB3", "#E8868C", "#E3AD5E", "#6FAAD8", "#A88EC8", "#72B88E"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', Arial, Helvetica, sans-serif;
    background-color: #FFFFFF;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F8F9FB !important;
    border-right: 1px solid #D1D5DB !important;
}
[data-testid="stSidebar"] * { color: #1C1E26 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.87rem !important;
    color: #5D6470 !important;
}
[data-testid="stSidebar"] hr { border-color: #D1D5DB !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1C1E26 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 500 !important;
    color: #5D6470 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Page titles ── */
h1 { color: #1C1E26 !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
h2 { color: #1C1E26 !important; font-weight: 600 !important; }
h3 { color: #5D6470 !important; font-weight: 600 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid #D1D5DB;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.87rem;
    color: #9CA3AF;
    padding: 8px 16px;
    border-radius: 4px 4px 0 0;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #1C1E26 !important;
    background: #F8F9FB !important;
    border-bottom: 2px solid #6FC3C8 !important;
    font-weight: 600 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #D1D5DB;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #1C1E26 !important;
    background: #F8F9FB !important;
    border-radius: 4px !important;
    border: 1px solid #E9ECEF !important;
}

/* ── Buttons ── */
.stDownloadButton button {
    background: #6FC3C8 !important;
    color: #1C1E26 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* ── Info / callouts ── */
.stInfo  { border-radius: 4px !important; background: #FFF8E8 !important; }
.stSuccess, .stWarning, .stError { border-radius: 4px !important; }

/* ── Divider ── */
hr { border-color: #E9ECEF !important; }
</style>
""", unsafe_allow_html=True)


def apply_template(fig, height=420, margin=None):
    """Plotly template matching magazine_style: white bg, dotted y-grid, pastel palette."""
    m = margin or dict(l=64, r=40, t=52, b=52)
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color=P_SLATE),
        plot_bgcolor=P_BG,
        paper_bgcolor=P_BG,
        margin=m,
        height=height,
        colorway=PALETTE,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color=P_SLATE),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True, linecolor=P_RULE, linewidth=0.8,
        tickfont=dict(size=10, color=P_SLATE),
        tickcolor=P_GHOST,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=P_GRID, gridwidth=0.7,
        griddash="dot",
        showline=True, linecolor=P_RULE, linewidth=0.8,
        tickfont=dict(size=10, color=P_SLATE),
        tickcolor=P_GHOST,
    )
    return fig


def kpi_card(title, value, subtitle="", color=P_AQUA):
    return f"""
    <div style="background:{P_BG};border:1px solid {P_RULE};
                border-left:3px solid {color};border-radius:4px;
                padding:16px 18px;height:100%">
      <div style="font-size:1.75rem;font-weight:700;color:{P_INK};line-height:1.1">{value}</div>
      <div style="font-weight:600;color:{P_INK};margin:5px 0 3px;font-size:0.88rem">{title}</div>
      <div style="color:{P_SLATE};font-size:0.78rem">{subtitle}</div>
    </div>"""


def finding_card(icon, title, value, description, color):
    return f"""
    <div style="background:{P_BG};border:1px solid {P_RULE};
                border-top:3px solid {color};border-radius:4px;
                padding:18px 20px;height:100%">
      <div style="font-size:1.25rem;font-weight:700;color:{P_INK};margin-bottom:2px">{value}</div>
      <div style="font-weight:600;color:{P_INK};margin-bottom:8px;font-size:0.88rem">{title}</div>
      <div style="color:{P_SLATE};font-size:0.82rem;line-height:1.6">{description}</div>
    </div>"""




# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
APP_DIR  = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


@st.cache_data
def load_data():
    scored = pd.read_csv(DATA_DIR / "scored_candidates.tsv", sep="\t")
    ref    = pd.read_csv(DATA_DIR / "reference_profile.tsv",  sep="\t")
    return scored, ref


@st.cache_data
def load_structural():
    try:
        cand  = pd.read_csv(DATA_DIR / "candidates_structural.tsv",    sep="\t")
        known = pd.read_csv(DATA_DIR / "known_ucoes_structural.tsv",   sep="\t")
        ctrl  = pd.read_csv(DATA_DIR / "controls_structural.tsv",      sep="\t")
        return cand, known, ctrl
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_integrated():
    path = DATA_DIR / "ucoe_integrated_classification.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return None


@st.cache_data
def load_validation():
    loo  = pd.read_csv(DATA_DIR / "loo_validation.tsv",       sep="\t")
    sens = pd.read_csv(DATA_DIR / "sensitivity_analysis.tsv", sep="\t")
    return loo, sens


@st.cache_data
def load_fasta_sequences():
    seqs = {}
    fasta_path = DATA_DIR / "ucoe_sequences.fa"
    if not fasta_path.exists():
        return seqs
    header, seq_lines = None, []
    for line in fasta_path.read_text().splitlines():
        if line.startswith(">"):
            if header and seq_lines:
                seqs[header] = "".join(seq_lines).upper()
            header = line[1:].split()[0]
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if header and seq_lines:
        seqs[header] = "".join(seq_lines).upper()
    return seqs


@st.cache_data
def load_conservation_top50():
    npz_path = DATA_DIR / "conservation_top50.npz"
    if npz_path.exists():
        return dict(np.load(npz_path, allow_pickle=True))
    return {}


@st.cache_data
def load_phastcons_top50():
    npz_path = DATA_DIR / "phastcons_top50.npz"
    if npz_path.exists():
        return dict(np.load(npz_path, allow_pickle=True))
    return {}


# ── Load everything ──
scored, ref              = load_data()
struct_cand, struct_known, struct_ctrl = load_structural()
integrated               = load_integrated()
loo_df, sens_raw         = load_validation()

# ── Derived columns ──
scored["length"] = scored["end"] - scored["start"]
scored["label"]  = scored["gene1"] + "/" + scored["gene2"]
scored["region"] = (scored["chrom"] + ":" +
                    scored["start"].astype(str) + "-" +
                    scored["end"].astype(str))

# ── Known UCOE labels ──
KNOWN_UCOES = {
    "A2UCOE":    ("CBX3",  "HNRNPA2B1"),
    "TBP/PSMB1": ("PSMB1", "TBP"),
    "SRF-UCOE":  ("SURF2", "SURF1"),
}

def is_known_ucoe(row):
    for name, (g1, g2) in KNOWN_UCOES.items():
        if ({row["gene1"], row["gene2"]} == {g1, g2}):
            return name
    return None

scored["known_ucoe"] = scored.apply(is_known_ucoe, axis=1)

# ── Merge sensitivity with gene names ──
sens = sens_raw.merge(
    scored[["region", "gene1", "gene2", "composite_rank", "composite_score", "label", "chrom"]],
    on="region", how="left"
)

# ── Merge integrated classification with scored ──
if integrated is not None:
    integrated["label"] = integrated["label"].str.strip()
    scored_full = scored.merge(
        integrated[["label", "n_ets", "ets_density_kb", "ets_in_nfr",
                    "frac_ets_in_nfr", "ww_fraction", "ss_fraction",
                    "phylop_mean", "ets_phylop_mean", "feature_score",
                    "dinuc_entropy", "nuc_score"]],
        on="label", how="left"
    )
else:
    scored_full = scored.copy()

# ── WW/SS reference values from actual data ──
if struct_ctrl is not None and len(struct_ctrl) > 0:
    REF_CTRL_WW = struct_ctrl["ww_fraction"].median()
    REF_CTRL_SS = struct_ctrl["ss_fraction"].median()
else:
    REF_CTRL_WW, REF_CTRL_SS = 0.081, 0.474

if struct_cand is not None and len(struct_cand) > 0:
    REF_CAND_WW = struct_cand["ww_fraction"].median()
    REF_CAND_SS = struct_cand["ss_fraction"].median()
else:
    REF_CAND_WW, REF_CAND_SS = 0.112, 0.431

# ── Features (21 variables — full pipeline feature space) ──
FEATURES = [
    "H3K4me3_mean", "H3K4me3_cv", "H3K27ac_mean", "H3K27ac_cv",
    "H3K27me3_mean", "H3K27me3_cv", "H3K9me3_mean", "H3K9me3_cv",
    "meth_mean", "meth_cv", "DNase_mean", "DNase_cv",
    "H3K9ac_mean", "H3K9ac_cv", "H3K36me3_mean", "H3K36me3_cv",
    "repliseq_mean", "CTCF_n_peaks",
    "cpg_obs_exp", "cpg_gc_pct", "inter_tss_distance",
]

FEATURE_LABELS = {
    "H3K4me3_mean": "H3K4me3",       "H3K4me3_cv": "H3K4me3 CV",
    "H3K27ac_mean": "H3K27ac",       "H3K27ac_cv": "H3K27ac CV",
    "H3K27me3_mean": "H3K27me3",     "H3K27me3_cv": "H3K27me3 CV",
    "H3K9me3_mean": "H3K9me3",       "H3K9me3_cv": "H3K9me3 CV",
    "meth_mean": "Methylation",       "meth_cv": "Meth CV",
    "DNase_mean": "DNase",            "DNase_cv": "DNase CV",
    "H3K9ac_mean": "H3K9ac",         "H3K9ac_cv": "H3K9ac CV",
    "H3K36me3_mean": "H3K36me3",     "H3K36me3_cv": "H3K36me3 CV",
    "repliseq_mean": "Repli-seq",     "CTCF_n_peaks": "CTCF peaks",
    "cpg_obs_exp": "CpG Obs/Exp",    "cpg_gc_pct": "GC %",
    "inter_tss_distance": "Inter-TSS dist",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(
    f"<div style='padding-bottom:14px;margin-bottom:14px;border-bottom:1px solid {P_RULE}'>"
    f"<div style='font-size:1.15rem;font-weight:700;color:{P_INK};letter-spacing:-0.01em'>UCOE Atlas</div>"
    f"<div style='font-size:0.72rem;color:{P_GHOST};margin-top:2px'>Human Genome · GRCh38</div>"
    f"</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Pages",
    ["Overview", "Candidate Explorer", "Candidate Detail",
     "PCA Explorer", "Validation & Robustness",
     "Methods & Glossary", "Downloads", "About"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"<div style='margin-top:28px;padding-top:14px;border-top:1px solid {P_RULE}'>"
    f"<div style='font-size:0.68rem;font-weight:600;color:{P_GHOST};letter-spacing:0.07em;"
    f"text-transform:uppercase;margin-bottom:8px'>Reference UCOEs</div>"
    f"<div style='font-size:0.8rem;color:{P_SLATE};line-height:2.1'>"
    f"A2UCOE &nbsp;<span style='color:{P_GHOST}'>rank 27</span><br>"
    f"TBP/PSMB1 &nbsp;<span style='color:{P_GHOST}'>rank 121</span><br>"
    f"SRF-UCOE &nbsp;<span style='color:{P_GHOST}'>rank 188</span>"
    f"</div>"
    f"<div style='margin-top:18px;font-size:0.71rem;color:{P_GHOST};line-height:1.7'>"
    f"E.R. Ostetti · A.M. Moro<br>T.M. Manieri<br>USP / Instituto Butantan"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("UCOE Atlas")
    st.markdown(
        f"<p style='font-size:1.05rem;color:{P_SLATE};margin-top:-8px;margin-bottom:28px'>"
        "A computational atlas of Ubiquitous Chromatin Opening Element candidates "
        "in the human genome — GRCh38 · ENCODE/Roadmap Epigenomics · GENCODE v44</p>",
        unsafe_allow_html=True,
    )

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates identified", "599")
    c2.metric("Known UCOEs recovered", "3 / 3")
    c3.metric("Weight-stable (top 20)", "7")
    c4.metric("Epigenomic features", "21")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Findings ──
    st.markdown(
        "<h2 style='margin-bottom:16px'>Key Findings</h2>",
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(finding_card(
            None, "ETS Motif Enrichment",
            "FE = 1.94 · q = 1.3×10⁻¹³",
            "The CGGAAG hexamer is significantly overrepresented at UCOE candidates "
            "relative to matched CpG island controls, implicating ETS-family "
            "transcription factors as core regulators of constitutive chromatin opening.",
            C_BLUE,
        ), unsafe_allow_html=True)
    with f2:
        st.markdown(finding_card(
            None, "Sequence-Intrinsic Nucleosome Barrier",
            "+38% WW dinucleotides",
            "Candidates show higher WW (AA·AT·TA·TT) content than CpG island controls "
            "(median 0.112 vs 0.081), reducing thermodynamic affinity for histone "
            "octamers independently of epigenetic state.",
            C_TEAL,
        ), unsafe_allow_html=True)
    with f3:
        st.markdown(finding_card(
            None, "Purifying Selection on ETS Motifs",
            "PhyloP 0.51 vs 0.40",
            "ETS motif positions show higher mean PhyloP conservation scores than "
            "flanking sequence, consistent with functional constraint on "
            "transcription factor binding sites across vertebrate evolution.",
            C_GREEN,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Pipeline funnel ──
    st.subheader("Phase I — Filtering Pipeline")

    filter_steps = [
        ("All bidirectional HKG promoters", 789,  0,   C_BLUE),
        ("+ CpG island overlap ≥ 40%",      692,  97,  C_TEAL),
        ("+ Active marks ubiquitous ≥ 80%",  647,  45,  "#0277BD"),
        ("+ Repressive marks absent ≥ 80%",  645,   2,  C_AMBER),
        ("+ Constitutive hypomethylation",   599,  46,  C_GREEN),
    ]

    fig_funnel = go.Figure()
    bar_colors = [s[3] for s in filter_steps]
    labels     = [s[0] for s in filter_steps]
    retained   = [s[1] for s in filter_steps]
    eliminated = [s[2] for s in filter_steps]

    fig_funnel.add_trace(go.Bar(
        y=labels, x=retained, orientation="h",
        marker_color=bar_colors, opacity=0.82,
        text=[f"  {r:,}" for r in retained],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=13, color="white", family="Inter, Arial, sans-serif"),
        hovertemplate="<b>%{y}</b><br>Retained: %{x:,}<extra></extra>",
        name="Retained",
    ))

    for i, (label, ret, elim, _) in enumerate(filter_steps):
        if elim > 0:
            fig_funnel.add_annotation(
                x=ret + 22, y=i,
                text=f"−{elim}",
                showarrow=False,
                font=dict(size=11, color=P_ROSE),
                xanchor="left",
            )

    apply_template(fig_funnel, height=280, margin=dict(l=280, r=80, t=20, b=40))
    fig_funnel.update_layout(showlegend=False, xaxis_title="Candidates retained")
    fig_funnel.update_xaxes(range=[0, 900], showticklabels=False, showgrid=False, showline=False)
    fig_funnel.update_yaxes(autorange="reversed", tickfont=dict(size=12, color=P_SLATE))
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # ── Score distribution ──
    col_hist, col_phase2 = st.columns([2, 1])

    with col_hist:
        st.subheader("Phase II — Composite Score Distribution")
        fig_dist = px.histogram(
            scored, x="composite_score", nbins=50,
            color_discrete_sequence=[C_BLUE],
            labels={"composite_score": "Composite Score"},
        )
        # Known UCOE markers
        for _, row in scored[scored["known_ucoe"].notna()].iterrows():
            fig_dist.add_vline(
                x=row["composite_score"], line_dash="dash", line_color=C_RED, line_width=2,
                annotation_text=f"<b>{row['known_ucoe']}</b>",
                annotation_position="top", annotation_font=dict(size=11, color=C_RED),
            )
        # Top-20 shading
        top20_thresh = scored.nsmallest(20, "composite_rank")["composite_score"].min()
        fig_dist.add_vrect(
            x0=top20_thresh, x1=scored["composite_score"].max() + 0.01,
            fillcolor=C_AMBER, opacity=0.07, line_width=0,
            annotation_text="Top 20", annotation_position="top left",
            annotation_font=dict(size=11, color=C_AMBER),
        )
        apply_template(fig_dist, height=320)
        fig_dist.update_layout(showlegend=False, yaxis_title="Candidates")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_phase2:
        st.subheader("Ranking Metrics")
        st.markdown(f"""
        <div style="background:{C_BGCARD};border:1px solid {C_BORDER};
                    border-radius:10px;padding:20px;margin-top:8px">
        <p style="margin:0 0 12px;font-weight:600;color:{C_NAVY}">Composite Score</p>
        <p style="color:#5D6470;font-size:0.88rem;line-height:1.7;margin:0">
        <b>S = 0.4 · D̃<sub>M</sub> + 0.3 · cos̃ + 0.3 · P̃</b><br><br>
        Each metric is min-max normalised to [0, 1].<br><br>
        <b style="color:{C_BLUE}">Mahalanobis (×0.4)</b><br>
        Distance to UCOE centroid with Ledoit-Wolf covariance.<br><br>
        <b style="color:{C_TEAL}">Cosine similarity (×0.3)</b><br>
        Angle to reference centroid in z-score space.<br><br>
        <b style="color:{C_GREEN}">Percentile rank (×0.3)</b><br>
        Feature-wise percentile, averaged over 21 features.
        </p></div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Candidate Explorer":
    st.title("Candidate Explorer")

    # ── Filters ──
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        rank_range = st.slider("Rank range", 1, 599, (1, 50))
    with col2:
        chrom_filter = st.multiselect(
            "Chromosome", sorted(scored["chrom"].unique()), default=[])
    with col3:
        min_score = st.slider("Min composite score", 0.0, 1.0, 0.0, 0.01)
    with col4:
        only_stable = st.checkbox("100% stable only")

    stable_labels = set(sens[sens["stability_pct"] == 100.0]["label"].dropna())

    mask = (
        (scored["composite_rank"] >= rank_range[0]) &
        (scored["composite_rank"] <= rank_range[1]) &
        (scored["composite_score"] >= min_score)
    )
    if chrom_filter:
        mask &= scored["chrom"].isin(chrom_filter)
    if only_stable:
        mask &= scored["label"].isin(stable_labels)

    filtered = scored_full[mask].sort_values("composite_rank")
    st.markdown(f"**{len(filtered)} candidates** matching filters")

    # ── Table ──
    display_cols = [
        "composite_rank", "label", "chrom", "length",
        "composite_score", "mahalanobis_dist", "cosine_sim",
        "H3K4me3_mean", "H3K27ac_mean", "meth_mean",
        "H3K27me3_mean", "H3K9me3_mean",
        "ets_density_kb", "ww_fraction", "known_ucoe",
    ]
    existing = [c for c in display_cols if c in filtered.columns]
    rename = {
        "composite_rank": "Rank", "label": "Gene Pair", "length": "Length (bp)",
        "composite_score": "Score", "mahalanobis_dist": "Mahal. Dist",
        "cosine_sim": "Cos. Sim", "H3K4me3_mean": "H3K4me3",
        "H3K27ac_mean": "H3K27ac", "meth_mean": "Meth %",
        "H3K27me3_mean": "H3K27me3", "H3K9me3_mean": "H3K9me3",
        "ets_density_kb": "ETS/kb", "ww_fraction": "WW frac.",
        "known_ucoe": "Known UCOE",
    }
    st.dataframe(
        filtered[existing].rename(columns=rename).round(3),
        use_container_width=True, hide_index=True, height=420,
    )

    st.markdown("---")
    col_chr, col_scatter = st.columns(2)

    with col_chr:
        st.subheader("Chromosome Distribution")
        chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        chrom_counts = (filtered["chrom"].value_counts()
                        .reindex(chrom_order).dropna().astype(int))
        fig_chr = go.Figure(go.Bar(
            x=chrom_counts.index, y=chrom_counts.values,
            marker_color=C_TEAL, opacity=0.85,
            hovertemplate="%{x}: %{y} candidates<extra></extra>",
        ))
        apply_template(fig_chr, height=300)
        fig_chr.update_layout(showlegend=False, yaxis_title="Candidates")
        st.plotly_chart(fig_chr, use_container_width=True)

    with col_scatter:
        st.subheader("Score vs ETS Density")
        if "ets_density_kb" in filtered.columns:
            scatter_df = filtered.dropna(subset=["ets_density_kb"]).copy()
            scatter_df["is_stable"] = scatter_df["label"].isin(stable_labels)
            scatter_df["is_known"]  = scatter_df["known_ucoe"].notna()

            fig_sc = px.scatter(
                scatter_df, x="ets_density_kb", y="composite_score",
                color="composite_rank",
                color_continuous_scale="Blues_r",
                hover_data=["label", "composite_rank", "chrom"],
                labels={"ets_density_kb": "ETS motifs / kb",
                        "composite_score": "Composite Score",
                        "composite_rank": "Rank"},
            )
            # Stable stars
            stable_sub = scatter_df[scatter_df["is_stable"]]
            if len(stable_sub):
                fig_sc.add_trace(go.Scatter(
                    x=stable_sub["ets_density_kb"], y=stable_sub["composite_score"],
                    mode="markers",
                    marker=dict(size=14, symbol="star", color=C_AMBER,
                                line=dict(width=1.5, color="white")),
                    name="100% stable",
                    hovertext=stable_sub["label"], hoverinfo="text",
                ))
            apply_template(fig_sc, height=300)
            fig_sc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("ETS data not available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Candidate Detail":
    import re as _re
    from scipy.ndimage import uniform_filter1d

    sorted_scored = scored.sort_values("composite_rank")
    stable_set    = set(sens[sens["stability_pct"] == 100]["label"].dropna())

    options = []
    for _, r in sorted_scored.iterrows():
        tag  = " ★"        if r["known_ucoe"]          else ""
        stab = " · stable" if r["label"] in stable_set else ""
        options.append(f"#{int(r['composite_rank'])} — {r['label']} ({r['chrom']}){tag}{stab}")

    known_opts  = [o for o in options if "★" in o]
    other_opts  = [o for o in options if "★" not in o]
    sel_col, _ = st.columns([2, 3])
    with sel_col:
        selected = st.selectbox("Candidate", known_opts + other_opts,
                                help="★ Known UCOEs listed first. · stable = 100% weight-stable.")

    rank = int(selected.split("#")[1].split(" ")[0])
    cand = scored_full[scored_full["composite_rank"] == rank].iloc[0]

    # ── Title block ──────────────────────────────────────────────────────────
    known    = cand["known_ucoe"]
    is_stable = cand["label"] in stable_set

    badges = ""
    if known:
        badges += (f"<span style='background:{P_AQUA};color:#fff;font-size:0.72rem;"
                   f"font-weight:600;padding:3px 9px;border-radius:3px;margin-right:6px'>"
                   f"Validated UCOE</span>")
    if is_stable:
        badges += (f"<span style='background:{P_MINT};color:#fff;font-size:0.72rem;"
                   f"font-weight:600;padding:3px 9px;border-radius:3px;margin-right:6px'>"
                   f"100% weight-stable</span>")

    stab_row   = sens[sens["label"] == cand["label"]]
    stab_pct   = f"{stab_row.iloc[0]['stability_pct']:.0f}%" if len(stab_row) else "—"
    has_int    = integrated is not None and cand["label"] in integrated["label"].values
    int_row    = integrated[integrated["label"] == cand["label"]].iloc[0] if has_int else None

    st.markdown(
        f"<div style='margin-bottom:4px'>{badges}</div>"
        f"<h1 style='font-size:1.7rem;font-weight:700;color:{P_INK};"
        f"letter-spacing:-0.02em;margin-bottom:2px'>{cand['label']}</h1>"
        f"<p style='color:{P_SLATE};font-size:0.9rem;margin-top:0'>"
        f"{cand['chrom']}:{cand['start']:,}–{cand['end']:,} &nbsp;·&nbsp; "
        f"{cand['length']:,} bp &nbsp;·&nbsp; "
        f"Inter-TSS {cand['inter_tss_distance']:,} bp &nbsp;·&nbsp; "
        f"Type: {cand.get('pair_type','N/A')}</p>",
        unsafe_allow_html=True,
    )

    # ── Score strip ──────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    sc1.metric("Rank",             f"#{int(cand['composite_rank'])}")
    sc2.metric("Composite Score",  f"{cand['composite_score']:.4f}")
    sc3.metric("Mahalanobis",      f"{cand['mahalanobis_dist']:.3f}")
    sc4.metric("Cosine Sim",       f"{cand['cosine_sim']:.3f}")
    sc5.metric("Weight Stability", stab_pct)
    sc6.metric("CpG Obs/Exp",      f"{cand['cpg_obs_exp']:.2f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # GENOMIC VIEWER  (UCSC-style multi-panel)
    # ══════════════════════════════════════════════════════════════════════════
    fasta_seqs        = load_fasta_sequences()
    conservation_data = load_conservation_top50()

    seq_key = None
    for key in fasta_seqs:
        if cand["gene1"] in key and cand["gene2"] in key:
            seq_key = key; break
        if str(cand["start"]) in key and str(cand["end"]) in key:
            seq_key = key; break

    if seq_key and seq_key in fasta_seqs:
        seq    = fasta_seqs[seq_key].upper()
        length = len(seq)
        WINDOW = 100

        # ETS motifs
        ets_fwd = [(m.start(), m.end(), "+", m.group()) for m in _re.finditer(r"CGGAA[GA]", seq)]
        ets_rev = [(m.start(), m.end(), "-", m.group()) for m in _re.finditer(r"[TC]TTCCG", seq)]
        ets_all = sorted(ets_fwd + ets_rev, key=lambda x: x[0])

        # TSS / NFR
        orig_start_rel = int(cand["orig_start"]) - int(cand["start"])
        orig_end_rel   = int(cand["orig_end"])   - int(cand["start"])
        tss1, tss2     = orig_start_rel, orig_end_rel
        nfr_regions    = [
            (max(0, tss1 - 200), min(length, tss1 + 200)),
            (max(0, tss2 - 200), min(length, tss2 + 200)),
        ]

        # GC% and CpG density — per-base convolution (matches generate_smim27 script)
        seq_pos     = np.arange(length)
        gc_arr      = np.array([1.0 if c in "GC" else 0.0 for c in seq])
        gc_vals     = np.convolve(gc_arr, np.ones(WINDOW) / WINDOW, mode="same") * 100
        cpg_arr     = np.array([1.0 if i < length - 1 and seq[i] == "C" and seq[i+1] == "G"
                                 else 0.0 for i in range(length)])
        cpg_vals    = np.convolve(cpg_arr, np.ones(WINDOW), mode="same")

        # PhyloP + PhastCons
        phastcons_data = load_phastcons_top50()
        cons_key       = f"{cand['gene1']}_{cand['gene2']}"
        has_cons       = cons_key in conservation_data
        has_phast      = cons_key in phastcons_data
        phylop_raw     = conservation_data.get(cons_key, None)
        phast_raw      = phastcons_data.get(cons_key, None)
        if has_cons:
            phylop_s = uniform_filter1d(np.nan_to_num(phylop_raw, nan=0.0), size=5)
            pos_full = np.arange(len(phylop_s))
        if has_phast:
            phast_s  = uniform_filter1d(np.nan_to_num(phast_raw,  nan=0.0), size=5)
            phast_pos = np.arange(len(phast_s))

        # Panel list and row heights
        # Fixed order: A gene struct | B PhyloP | C PhastCons | D GC%+CpG
        panels = [("A", "Gene structure", 0.22)]
        if has_cons:
            panels.append(("B", "PhyloP (100 vertebrates)", 0.26))
        if has_phast:
            panels.append((panels[-1][0] if not has_cons else "C", "PhastCons (100 vertebrates)", 0.22))
        # Relabel after all panels known
        letters = "ABCD"
        panels  = [(letters[i], p[1], p[2]) for i, p in enumerate(panels)]
        panels.append((letters[len(panels)], "GC% & CpG density", 0.30))

        n_rows = len(panels)
        row_h  = [p[2] for p in panels]
        # Normalise
        total  = sum(row_h)
        row_h  = [h / total for h in row_h]

        n_rows  = len(panel_names)
        fig_ucsc = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            row_heights=row_h, vertical_spacing=0.02,
        )

        def _shade_ets(fig, ets_all, row):
            for s, e, _, _ in ets_all:
                fig.add_vrect(x0=s, x1=e,
                              fillcolor=f"rgba(246,196,122,0.38)", line_width=0,
                              row=row, col=1)

        cur = 1   # current row counter

        # ── Panel A: Gene structure (UCSC-style two-track layout) ────────────
        # Three y-levels:  gene1 track = +0.7,  ETS track = 0,  gene2 track = -0.7
        # Each gene occupies its own horizontal lane; no visual overlap.

        Y_G1, Y_G2, Y_ETS = 0.70, -0.70, 0.0
        TRACK_H = 0.18   # half-height of the gene body bar

        # ── NFR shading (only in the promoter overlap zone) ──
        nfr_rgba = ["rgba(111,195,200,0.18)", "rgba(242,154,160,0.18)"]
        nfr_y    = [(Y_G1 - TRACK_H, Y_G1 + TRACK_H),
                    (Y_G2 - TRACK_H, Y_G2 + TRACK_H)]
        for (ns, ne), nfc, (ny0, ny1) in zip(nfr_regions, nfr_rgba, nfr_y):
            fig_ucsc.add_shape(
                type="rect", x0=ns, x1=ne, y0=ny0 - 0.08, y1=ny1 + 0.08,
                fillcolor=nfc, line=dict(width=0), row=cur, col=1,
            )

        # ── TSS dotted verticals ──
        for tss_p, ty in [(tss1, Y_G1), (tss2, Y_G2)]:
            fig_ucsc.add_shape(
                type="line", x0=tss_p, x1=tss_p, y0=-1.1, y1=1.1,
                line=dict(color=P_RULE, width=1, dash="dot"), row=cur, col=1,
            )
            fig_ucsc.add_annotation(
                x=tss_p, y=(1.12 if ty > 0 else -1.12),
                text="TSS", showarrow=False,
                font=dict(size=8, color=P_GHOST), row=cur, col=1,
            )

        # ── Gene body bars + directional arrow tip ──
        g1_x0 = max(0, tss1)
        g1_x1 = length - 1
        g2_x0 = 0
        g2_x1 = min(length - 1, tss2)

        for gx0, gx1, gy, gc, gname, direction in [
            (g1_x0, g1_x1, Y_G1, P_AQUA, cand["gene1"], "+"),
            (g2_x0, g2_x1, Y_G2, P_ROSE, cand["gene2"], "-"),
        ]:
            # Gene body (thin filled rect)
            fig_ucsc.add_shape(
                type="rect", x0=gx0, x1=gx1,
                y0=gy - TRACK_H, y1=gy + TRACK_H,
                fillcolor=gc, opacity=0.82,
                line=dict(width=0), row=cur, col=1,
            )
            # Arrowhead at 3′ end
            arrow_x = gx1 if direction == "+" else gx0
            ax_offset = length * 0.012
            fig_ucsc.add_annotation(
                x=arrow_x + (ax_offset if direction == "+" else -ax_offset),
                y=gy, ax=arrow_x, ay=gy,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True,
                arrowhead=2, arrowsize=1.4, arrowwidth=2.5, arrowcolor=gc,
                row=cur, col=1,
            )
            # Gene name centred on body
            fig_ucsc.add_annotation(
                x=(gx0 + gx1) / 2, y=gy + (TRACK_H + 0.14),
                text=f"<i><b>{gname}</b></i>",
                showarrow=False,
                font=dict(size=11, color=gc, family="Arial"),
                row=cur, col=1,
            )
            # Direction label at TSS end
            dir_label = "5′ →" if direction == "+" else "← 5′"
            dir_x     = gx0 if direction == "+" else gx1
            fig_ucsc.add_annotation(
                x=dir_x, y=gy - (TRACK_H + 0.18),
                text=dir_label, showarrow=False,
                font=dict(size=8, color=gc), row=cur, col=1,
            )

        # ── ETS markers on centre track ──
        for i, (s, e, strand, mseq) in enumerate(ets_all):
            in_nfr = any(ns <= s <= ne for ns, ne in nfr_regions)
            sym    = "triangle-down" if strand == "+" else "triangle-up"
            fig_ucsc.add_trace(go.Scatter(
                x=[(s + e) / 2], y=[Y_ETS], mode="markers",
                marker=dict(size=10, color=P_PEACH, symbol=sym,
                            line=dict(width=1.2, color=P_INK)),
                hovertemplate=(f"ETS #{i+1} ({strand})<br>{mseq}<br>"
                               f"pos {s}<br>{'in NFR' if in_nfr else 'outside NFR'}"
                               f"<extra></extra>"),
                showlegend=False,
            ), row=cur, col=1)
            fig_ucsc.add_annotation(
                x=(s + e) / 2, y=Y_ETS + (0.22 if strand == "+" else -0.22),
                text=f"#{i+1}", showarrow=False,
                font=dict(size=7, color=P_SLATE), row=cur, col=1,
            )

        fig_ucsc.update_yaxes(
            visible=False, range=[-1.2, 1.2], row=cur, col=1,
        )
        cur += 1

        # ── Panel B: PhyloP ──────────────────────────────────────────────────
        if has_cons:
            _shade_ets(fig_ucsc, ets_all, cur)
            fig_ucsc.add_trace(go.Scatter(
                x=pos_full, y=np.maximum(phylop_s, 0),
                mode="lines", name="Conserved",
                line=dict(color=P_AQUA, width=0.6),
                fill="tozeroy", fillcolor="rgba(111,195,200,0.70)",
            ), row=cur, col=1)
            fig_ucsc.add_trace(go.Scatter(
                x=pos_full, y=np.minimum(phylop_s, 0),
                mode="lines", name="Accelerated",
                line=dict(color=P_ROSE, width=0.6),
                fill="tozeroy", fillcolor="rgba(242,154,160,0.60)",
            ), row=cur, col=1)
            fig_ucsc.add_hline(y=0, line_color=P_GHOST, line_width=0.8,
                               line_dash="dash", row=cur, col=1)
            fig_ucsc.update_yaxes(
                title_text="PhyloP", title_font=dict(size=9, color=P_SLATE),
                zeroline=False, row=cur, col=1,
            )
            cur += 1

        # ── Panel C: PhastCons ───────────────────────────────────────────────
        if has_phast:
            _shade_ets(fig_ucsc, ets_all, cur)
            fig_ucsc.add_trace(go.Scatter(
                x=phast_pos, y=phast_s,
                mode="lines", name="PhastCons",
                line=dict(color=P_MINT, width=0.8),
                fill="tozeroy", fillcolor="rgba(143,207,168,0.65)",
            ), row=cur, col=1)
            fig_ucsc.update_yaxes(
                title_text="PhastCons", title_font=dict(size=9, color=P_SLATE),
                range=[0, 1.05], row=cur, col=1,
            )
            cur += 1

        # ── Panel D: GC% + CpG density ───────────────────────────────────────
        _shade_ets(fig_ucsc, ets_all, cur)
        fig_ucsc.add_trace(go.Scatter(
            x=seq_pos, y=gc_vals, mode="lines", name="GC %",
            line=dict(color=P_SKY, width=1.8),
            fill="tozeroy", fillcolor="rgba(136,191,235,0.22)",
        ), row=cur, col=1)
        fig_ucsc.add_hline(y=50, line_color=P_GHOST, line_width=0.7,
                           line_dash="dot", row=cur, col=1)
        cpg_max    = float(cpg_vals.max()) if cpg_vals.max() > 0 else 1.0
        cpg_scaled = cpg_vals / cpg_max * 95
        fig_ucsc.add_trace(go.Bar(
            x=seq_pos, y=cpg_scaled, name="CpG / window",
            marker_color="rgba(192,168,220,0.62)", width=1,
        ), row=cur, col=1)
        fig_ucsc.update_yaxes(
            title_text="GC %", title_font=dict(size=9, color=P_SLATE),
            range=[0, 100], row=cur, col=1,
        )

        # ── Shared layout ─────────────────────────────────────────────────────
        fig_ucsc.update_xaxes(
            showgrid=False, showline=True, linecolor=P_RULE, linewidth=0.8,
            tickfont=dict(size=10, color=P_SLATE), tickcolor=P_GHOST,
        )
        fig_ucsc.update_xaxes(
            title_text=f"Position in {cand['label']} locus (bp)",
            title_font=dict(size=11, color=P_SLATE),
            row=cur, col=1,
        )
        fig_ucsc.update_layout(
            height=130 * n_rows + 80,
            showlegend=False,
            margin=dict(l=72, r=28, t=28, b=44),
            paper_bgcolor=P_BG, plot_bgcolor=P_BG,
            font=dict(family="Inter, Arial, sans-serif", size=11, color=P_SLATE),
        )
        # Panel labels (A, B, C, D) — match figure style
        panel_letters = "ABCD"
        row_positions  = [1 - (i + 0.5) / n_rows for i in range(n_rows)]
        for i, (letter, yp) in enumerate(zip(panel_letters[:n_rows], row_positions)):
            fig_ucsc.add_annotation(
                x=-0.04, y=yp, xref="paper", yref="paper",
                text=f"<b>{letter}</b>",
                showarrow=False, font=dict(size=13, color=P_INK),
                xanchor="right", yanchor="middle",
            )

        st.plotly_chart(fig_ucsc, use_container_width=True)

        # ── Legend ────────────────────────────────────────────────────────────
        def _leg_item(color, label, shape="rect"):
            if shape == "triangle":
                icon = (f"<span style='display:inline-block;width:0;height:0;"
                        f"border-left:6px solid transparent;border-right:6px solid transparent;"
                        f"border-top:10px solid {color};vertical-align:middle;"
                        f"margin-right:5px'></span>")
            else:
                icon = (f"<span style='display:inline-block;width:12px;height:10px;"
                        f"background:{color};border-radius:2px;vertical-align:middle;"
                        f"margin-right:5px;opacity:0.82'></span>")
            return f"{icon}<span style='font-size:0.78rem;color:{P_SLATE}'>{label}</span>"

        legend_items = [
            _leg_item(P_AQUA,     f"{cand['gene1']} (+)"),
            _leg_item(P_ROSE,     f"{cand['gene2']} (−)"),
            _leg_item(P_PEACH,    "ETS motif (CGGAA[GA])", "triangle"),
            _leg_item("rgba(246,196,122,0.55)", "ETS position (all panels)"),
        ]
        if has_cons:
            legend_items += [
                _leg_item("rgba(111,195,200,0.82)", "PhyloP > 0 — conserved"),
                _leg_item("rgba(242,154,160,0.82)", "PhyloP < 0 — accelerated"),
            ]
        if has_phast:
            legend_items.append(_leg_item("rgba(143,207,168,0.82)", "PhastCons probability"))
        legend_items += [
            _leg_item(P_SKY,      "GC content (%)"),
            _leg_item(P_LAVENDER, "CpG dinucleotide density"),
        ]

        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;gap:16px;padding:8px 4px;"
            f"border-top:1px solid {P_RULE};margin-top:4px'>"
            + "".join(f"<div>{item}</div>" for item in legend_items)
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── ETS summary table ─────────────────────────────────────────────────
        if ets_all:
            n_in_nfr = sum(1 for s, e, _, _ in ets_all
                           if any(ns <= s <= ne for ns, ne in nfr_regions))
            st.caption(
                f"{len(ets_all)} ETS motifs · "
                f"{len(ets_all)/length*1000:.2f} motifs/kb · "
                f"{n_in_nfr} within ±200 bp of TSS (NFR)"
            )
            with st.expander("ETS motif table"):
                ets_df = pd.DataFrame([
                    {"#": i+1, "Position (bp)": s, "Strand": strand, "Motif": ms,
                     "In NFR": "Yes" if any(ns <= s <= ne for ns, ne in nfr_regions) else "No",
                     "Genomic coord.": f"{cand['chrom']}:{cand['start']+s:,}"}
                    for i, (s, e, strand, ms) in enumerate(ets_all)
                ])
                st.dataframe(ets_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Sequence not available for this candidate.")

    # ══════════════════════════════════════════════════════════════════════════
    # EPIGENOMIC PROFILE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.subheader("Epigenomic Profile")

    col_radar, col_repr = st.columns(2)

    with col_radar:
        features_radar = ["H3K4me3_mean", "H3K27ac_mean", "H3K9ac_mean",
                          "H3K36me3_mean", "DNase_mean", "repliseq_mean"]
        labels_radar   = ["H3K4me3", "H3K27ac", "H3K9ac", "H3K36me3", "DNase", "Repli-seq"]
        vals_cand, vals_ref = [], []
        for f in features_radar:
            fmin = scored[f].min(); fmax = scored[f].max()
            rng  = fmax - fmin if fmax != fmin else 1
            vals_cand.append((cand[f] - fmin) / rng)
            vals_ref.append((ref[f].mean() - fmin) / rng)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_cand + [vals_cand[0]], theta=labels_radar + [labels_radar[0]],
            fill="toself", name=cand["label"],
            line_color=P_AQUA, fillcolor="rgba(111,195,200,0.19)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_ref + [vals_ref[0]], theta=labels_radar + [labels_radar[0]],
            fill="toself", name="UCOE Reference",
            line_color=P_ROSE, fillcolor="rgba(242,154,160,0.09)", line_dash="dot",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                tickfont=dict(size=8, color=P_GHOST)),
                angularaxis=dict(tickfont=dict(size=11, color=P_SLATE)),
                bgcolor=P_BG,
            ),
            title=dict(text="Active Marks (min-max normalised)",
                       font=dict(size=12, color=P_INK), x=0.5),
            height=340,
            margin=dict(l=50, r=50, t=55, b=30),
            legend=dict(x=0.5, y=-0.08, xanchor="center", orientation="h",
                        font=dict(size=10, color=P_SLATE)),
            paper_bgcolor=P_BG,
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_repr:
        marks = {
            "H3K27me3": cand["H3K27me3_mean"],
            "H3K9me3":  cand["H3K9me3_mean"],
            "Methylation": cand["meth_mean"],
        }
        ref_marks = {
            "H3K27me3":    ref["H3K27me3_mean"].mean(),
            "H3K9me3":     ref["H3K9me3_mean"].mean(),
            "Methylation": ref["meth_mean"].mean(),
        }
        fig_repr = go.Figure()
        fig_repr.add_trace(go.Bar(
            x=list(marks.keys()), y=list(marks.values()),
            name=cand["label"], marker_color=P_AQUA, opacity=0.82,
        ))
        fig_repr.add_trace(go.Bar(
            x=list(ref_marks.keys()), y=list(ref_marks.values()),
            name="UCOE Reference", marker_color=P_ROSE, opacity=0.72,
        ))
        apply_template(fig_repr, height=340)
        fig_repr.update_layout(
            barmode="group",
            title=dict(text="Repressive Marks & Methylation",
                       font=dict(size=12, color=P_INK)),
            yaxis_title="Signal / %",
        )
        st.plotly_chart(fig_repr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DINUCLEOTIDE & BIOPHYSICAL
    # ══════════════════════════════════════════════════════════════════════════
    struct_cand, struct_known, struct_ctrl = load_structural()
    if struct_cand is not None:
        struct_row = None
        for _, sr in struct_cand.iterrows():
            h = str(sr.get("header", ""))
            if cand["gene1"] in h and cand["gene2"] in h:
                struct_row = sr; break

        if struct_row is not None:
            ww_val      = struct_row["ww_fraction"]
            ss_val      = struct_row["ss_fraction"]
            entropy_val = struct_row["dinuc_entropy"]

            with st.expander("Dinucleotide Composition & Biophysical Assessment"):
                st.caption(
                    "WW dinucleotides (AA, AT, TA, TT) reduce histone octamer affinity. "
                    "SS (CC, CG, GC, GG) stabilise nucleosomes. "
                    "Bars show this candidate vs. UCOE-candidate median and CpG-island controls."
                )
                fig_bio = make_subplots(
                    rows=1, cols=2, horizontal_spacing=0.12,
                    subplot_titles=("WW Fraction", "SS Fraction"),
                )
                for col_i, (val, ref_v, ctrl_v, color, label) in enumerate([
                    (ww_val, REF_CAND_WW, REF_CTRL_WW, P_ROSE,  "WW"),
                    (ss_val, REF_CAND_SS, REF_CTRL_SS, P_AQUA, "SS"),
                ], 1):
                    cats   = ["This candidate", "UCOE median", "Controls"]
                    values = [val, ref_v, ctrl_v]
                    _alpha = {"WW": ("rgba(242,154,160,1)", "rgba(242,154,160,0.55)", "rgba(156,163,175,0.55)"),
                              "SS": ("rgba(111,195,200,1)", "rgba(111,195,200,0.55)", "rgba(156,163,175,0.55)")}
                    clrs   = list(_alpha[label])
                    fig_bio.add_trace(go.Bar(
                        x=cats, y=values, marker_color=clrs, opacity=0.85,
                        text=[f"{v:.3f}" for v in values],
                        textposition="outside",
                        textfont=dict(size=11, color=P_INK),
                        showlegend=False,
                    ), row=1, col=col_i)
                apply_template(fig_bio, height=280)
                fig_bio.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                fig_bio.update_yaxes(range=[0, max(ww_val, ss_val, REF_CTRL_WW, REF_CTRL_SS) * 1.35])
                st.plotly_chart(fig_bio, use_container_width=True)

                # Assessment badges
                assessments = []
                assessments.append(("WW",
                    ("FAVORABLE", P_MINT) if ww_val >= REF_CAND_WW else
                    ("MODERATE", P_PEACH) if ww_val >= REF_CTRL_WW else ("LOW", P_ROSE)))
                assessments.append(("SS depletion",
                    ("FAVORABLE", P_MINT) if ss_val <= REF_CAND_SS else
                    ("MODERATE", P_PEACH) if ss_val <= REF_CTRL_SS else ("LOW", P_ROSE)))
                assessments.append(("Entropy",
                    ("FAVORABLE", P_MINT) if entropy_val >= 3.55 else ("MODERATE", P_PEACH)))

                ba_cols = st.columns(3)
                for (crit, (status, color)), col in zip(assessments, ba_cols):
                    col.markdown(
                        f"<div style='border:1px solid {P_RULE};border-top:3px solid {color};"
                        f"border-radius:4px;padding:10px 12px;text-align:center'>"
                        f"<div style='font-size:0.72rem;color:{P_GHOST};text-transform:uppercase;"
                        f"letter-spacing:0.05em;margin-bottom:4px'>{crit}</div>"
                        f"<div style='font-weight:700;color:{color};font-size:0.88rem'>{status}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Full dinucleotide spectrum"):
                    dinuc_cols   = [f"dinuc_{d}" for d in
                                    ["AA","AC","AG","AT","CA","CC","CG","CT",
                                     "GA","GC","GG","GT","TA","TC","TG","TT"]]
                    dinuc_labels = [d.replace("dinuc_", "") for d in dinuc_cols]
                    vals_this    = [struct_row[c] for c in dinuc_cols]
                    ww_set = {"AA","AT","TA","TT"}; ss_set = {"CC","CG","GC","GG"}
                    bar_colors_dn = [P_ROSE if d in ww_set else P_AQUA if d in ss_set
                                     else P_GHOST for d in dinuc_labels]
                    fig_dn = go.Figure()
                    fig_dn.add_trace(go.Bar(
                        x=dinuc_labels, y=vals_this, name="This candidate",
                        marker_color=bar_colors_dn, opacity=0.80,
                    ))
                    if struct_ctrl is not None:
                        ctrl_meds = [struct_ctrl[c].median() for c in dinuc_cols]
                        fig_dn.add_trace(go.Scatter(
                            x=dinuc_labels, y=ctrl_meds, mode="markers+lines",
                            name="Controls (median)",
                            line=dict(color=P_GHOST, width=1.5, dash="dash"),
                            marker=dict(size=5),
                        ))
                    if struct_known is not None:
                        known_means = [struct_known[c].mean() for c in dinuc_cols]
                        fig_dn.add_trace(go.Scatter(
                            x=dinuc_labels, y=known_means, mode="markers+lines",
                            name="Known UCOEs (mean)",
                            line=dict(color=P_PEACH, width=2),
                            marker=dict(size=7, symbol="diamond"),
                        ))
                    apply_template(fig_dn, height=280)
                    fig_dn.update_layout(yaxis_title="Proportion")
                    st.plotly_chart(fig_dn, use_container_width=True)
                    st.caption(
                        f"WW (rose) = AA, AT, TA, TT — disfavour nucleosomes. "
                        f"SS (aqua) = CC, CG, GC, GG — favour nucleosomes."
                    )

    # ── UCSC link ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ucsc_url = (f"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&"
                f"position={cand['chrom']}%3A{cand['start']}-{cand['end']}")
    st.markdown(
        f"<a href='{ucsc_url}' target='_blank' style='color:{P_AQUA};font-size:0.85rem'>"
        f"View locus in UCSC Genome Browser ↗</a>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PCA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "PCA Explorer":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    st.title("PCA Explorer — Epigenomic Feature Space")
    st.markdown(
        "<p style='color:#5D6470'>Principal component analysis of the 21-variable epigenomic "
        "feature space. Each point is one of the 599 candidates.</p>",
        unsafe_allow_html=True,
    )

    X = scored_full[FEATURES].copy()
    for col in FEATURES:
        if col in X.columns and X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    valid_feat = [f for f in FEATURES if f in X.columns]
    Z     = StandardScaler().fit_transform(X[valid_feat])
    pca   = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    pca_df = scored_full.copy()
    pca_df["PC1"] = Z_pca[:, 0]
    pca_df["PC2"] = Z_pca[:, 1]
    pca_df["stability_pct"] = pca_df["label"].map(
        sens.set_index("label")["stability_pct"]).fillna(0)

    col_opt, col_pca = st.columns([1, 3])
    with col_opt:
        color_by = st.selectbox("Colour by", [
            "composite_score", "composite_rank", "stability_pct",
            "H3K4me3_mean", "meth_mean", "length",
        ])
        show_known  = st.checkbox("Highlight known UCOEs", value=True)
        show_stable = st.checkbox("Highlight 100% stable", value=True)

    scale = "Viridis_r" if color_by in ("composite_rank",) else "Viridis"

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color=color_by,
        hover_data=["label", "composite_rank", "composite_score", "chrom"],
        color_continuous_scale=scale,
        opacity=0.65,
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        },
        title=f"599 UCOE Candidates — coloured by {color_by}",
    )

    if show_stable:
        stable_pca = pca_df[pca_df["stability_pct"] == 100]
        fig_pca.add_trace(go.Scatter(
            x=stable_pca["PC1"], y=stable_pca["PC2"],
            mode="markers", name="100% stable",
            marker=dict(size=13, symbol="star", color=C_AMBER,
                        line=dict(width=1.5, color="white")),
            hovertext=stable_pca["label"], hoverinfo="text+name",
        ))

    if show_known:
        known_pca = pca_df[pca_df["known_ucoe"].notna()]
        fig_pca.add_trace(go.Scatter(
            x=known_pca["PC1"], y=known_pca["PC2"],
            mode="markers+text",
            marker=dict(size=16, symbol="star-diamond", color=C_RED,
                        line=dict(width=2, color="white")),
            text=known_pca["known_ucoe"], textposition="top center",
            textfont=dict(size=11, color=C_RED, family="Inter"),
            name="Known UCOEs",
        ))

    apply_template(fig_pca, height=580)
    with col_pca:
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption(
            f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}% + "
            f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}% = "
            f"{sum(pca.explained_variance_ratio_)*100:.1f}% of variance explained. "
            f"⭐ = 100% weight-stable candidates. ✦ = known UCOEs."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VALIDATION & ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Validation & Robustness":
    st.title("Validation & Robustness")
    st.markdown(
        "<p style='color:#5D6470;font-size:1.05rem'>"
        "Two independent analyses assess whether the composite score is "
        "reliable and stable across methodological choices.</p>",
        unsafe_allow_html=True,
    )

    tab_loo, tab_sens = st.tabs(["Leave-One-Out Cross-Validation", "Weight Sensitivity Analysis"])

    # ── TAB 1: LOO ──
    with tab_loo:
        st.markdown("""
        ### Leave-One-Out Cross-Validation

        Each of the 3 known UCOEs is held out in turn. The pipeline is re-run
        using only the remaining 2 as reference, and the held-out UCOE is ranked
        against the 599 candidates.
        """)

        st.info("""
        **Interpreting LOO results with n = 3 references.** When the reference set
        contains only 3 elements, holding out one shifts the centroid substantially —
        the remaining pair may not adequately represent the full UCOE epigenomic
        space. Low LOO ranks therefore reflect small-sample instability of the
        reference centroid, not a failure of the scoring approach. This is a known
        limitation of methods that require a reference class and is addressed
        in the Methods section.
        """)

        # Full-model ranks for known UCOEs
        full_ranks = {
            row["label"]: (row["composite_rank"], row["composite_score"])
            for _, row in scored.iterrows() if row["known_ucoe"]
        }
        ucoe_name_map = {
            "A2UCOE_HNRNPA2B1_CBX3": "A2UCOE",
            "TBP_PSMB1":              "TBP/PSMB1",
            "SRF_UCOE_SURF1_SURF2":   "SRF-UCOE",
        }
        label_map = {
            "A2UCOE":    "CBX3/HNRNPA2B1",
            "TBP/PSMB1": "PSMB1/TBP",
            "SRF-UCOE":  "SURF2/SURF1",
        }

        loo_rows = []
        for _, row in loo_df.iterrows():
            ucoe_name  = ucoe_name_map.get(row["excluded_ucoe"], row["excluded_ucoe"])
            gene_label = label_map.get(ucoe_name, ucoe_name)
            fm = full_ranks.get(gene_label, (None, None))
            loo_rows.append({
                "Known UCOE":          ucoe_name,
                "Gene pair":           gene_label,
                "Rank — full model":   int(fm[0]) if fm[0] else "—",
                "Score — full model":  f"{fm[1]:.4f}" if fm[1] else "—",
                "Rank — LOO":          int(row["rank"]),
                "Score — LOO":         f"{row['score']:.4f}",
                "Top 10 (LOO)":        "✓" if row["top_10"] else "✗",
                "Top 50 (LOO)":        "✓" if row["top_50"] else "✗",
            })
        loo_display = pd.DataFrame(loo_rows)
        st.dataframe(loo_display, use_container_width=True, hide_index=True)

        # Paired bar chart: LOO rank vs full-model rank
        fig_loo = go.Figure()
        ucoe_names  = [r["Known UCOE"] for r in loo_rows]
        full_ranks_v = [r["Rank — full model"] for r in loo_rows]
        loo_ranks_v  = [r["Rank — LOO"]        for r in loo_rows]

        fig_loo.add_trace(go.Bar(
            name="Full model", x=ucoe_names, y=full_ranks_v,
            marker_color=C_BLUE, opacity=0.85,
            text=[f"#{v}" for v in full_ranks_v], textposition="outside",
        ))
        fig_loo.add_trace(go.Bar(
            name="LOO", x=ucoe_names, y=loo_ranks_v,
            marker_color=C_AMBER, opacity=0.75,
            text=[f"#{v}" for v in loo_ranks_v], textposition="outside",
        ))
        apply_template(fig_loo, height=380)
        fig_loo.update_layout(
            barmode="group",
            title="Known UCOE rank: full model vs leave-one-out",
            yaxis_title="Rank (lower = better)",
            yaxis=dict(autorange="reversed"),
        )
        fig_loo.add_hline(y=20,  line_dash="dot", line_color=C_GREEN,
                          annotation_text="Top 20",  annotation_position="right",
                          annotation_font=dict(size=10, color=C_GREEN))
        fig_loo.add_hline(y=599, line_dash="dot", line_color=C_GRAY,
                          opacity=0.3)
        st.plotly_chart(fig_loo, use_container_width=True)

        st.markdown(f"""
        **Interpretation.** In the full model (all 3 reference UCOEs), the known
        UCOEs rank at positions **27, 121 and 188** out of 599 — placing all three
        in the **top 32%** of candidates. The LOO experiment reveals that each known
        UCOE has a sufficiently distinctive profile that it is not well-approximated
        by the centroid of the remaining pair, which is expected given n = 3.
        Expansion of the validated UCOE reference set will be critical for improving
        LOO performance in future iterations.
        """)

    # ── TAB 2: SENSITIVITY ──
    with tab_sens:
        st.markdown("""
        ### Weight Sensitivity Analysis

        The composite score weights (w_Mahal, w_Cosine, w_Pct) were varied
        systematically over **29 combinations** spanning the weight simplex
        (w_M + w_C + w_P = 1, each ≥ 0.05, step 0.1). A candidate is classified
        as **stable** if it appears in the top 20 across all combinations.
        """)

        n_stable = int((sens["stability_pct"] == 100).sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("Weight combinations tested", "29")
        s2.metric("Candidates with 100% stability", str(n_stable))
        s3.metric("Candidates with > 80% stability",
                  str(int((sens["stability_pct"] >= 80).sum())))

        st.markdown("<br>", unsafe_allow_html=True)

        # Bubble chart
        fig_sens = go.Figure()

        # All candidates with stability data
        non_stable = sens[sens["stability_pct"] < 100].dropna(subset=["composite_rank"])
        stable_100 = sens[sens["stability_pct"] == 100].dropna(subset=["composite_rank"])

        if len(non_stable):
            fig_sens.add_trace(go.Scatter(
                x=non_stable["composite_rank"],
                y=non_stable["stability_pct"],
                mode="markers",
                marker=dict(
                    size=non_stable["composite_score"] * 18,
                    color=non_stable["stability_pct"],
                    colorscale=[[0, "#E3E8F0"], [0.5, C_TEAL], [1.0, C_BLUE]],
                    cmin=0, cmax=100,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Rank: %{x}<br>Stability: %{y:.1f}%<extra></extra>"
                ),
                customdata=non_stable[["label"]],
                name="Candidates",
                showlegend=False,
            ))

        if len(stable_100):
            fig_sens.add_trace(go.Scatter(
                x=stable_100["composite_rank"],
                y=stable_100["stability_pct"],
                mode="markers+text",
                marker=dict(
                    size=stable_100["composite_score"] * 22,
                    color=C_AMBER,
                    symbol="star",
                    line=dict(width=1.5, color="white"),
                ),
                text=stable_100["label"],
                textposition="top center",
                textfont=dict(size=9, color=C_NAVY, family="Inter"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Rank: %{x}<br>Stability: 100%<extra></extra>"
                ),
                name="100% stable ⭐",
            ))

        apply_template(fig_sens, height=480)
        fig_sens.update_layout(
            title="Rank stability across 29 weight combinations",
            xaxis_title="Composite rank (full model)",
            yaxis_title="Stability (% of combinations in top 20)",
            yaxis=dict(range=[-5, 108]),
        )
        fig_sens.add_hline(y=80, line_dash="dot", line_color=C_TEAL, opacity=0.6,
                           annotation_text=">80% stable",
                           annotation_position="right",
                           annotation_font=dict(size=10, color=C_TEAL))
        st.plotly_chart(fig_sens, use_container_width=True)

        # Top stable table
        st.subheader("Top 15 candidates by stability")
        top15 = (sens.dropna(subset=["composite_rank"])
                     .sort_values(["stability_pct", "composite_rank"],
                                  ascending=[False, True])
                     .head(15))
        top15_disp = top15[["label", "chrom", "composite_rank",
                             "composite_score", "stability_pct"]].copy()
        top15_disp.columns = ["Gene pair", "Chrom", "Rank", "Score", "Stability (%)"]
        top15_disp["Stability (%)"] = top15_disp["Stability (%)"].round(1)
        top15_disp["Score"] = top15_disp["Score"].round(4)
        st.dataframe(top15_disp, use_container_width=True, hide_index=True)

        st.markdown(f"""
        **Interpretation.** Seven candidates maintain a top-20 position
        across **all 29 weight combinations**, demonstrating that their high ranking
        is not an artefact of a specific weighting scheme. SMIM27/TOPORS (rank 1)
        is among them, supporting its designation as the top UCOE candidate
        regardless of how the three metrics are balanced.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METHODS & GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Methods & Glossary":
    st.title("Methods & Glossary")
    st.markdown(
        "<p style='color:#5D6470'>Technical description of the pipeline, scoring metrics, "
        "and epigenomic features used in candidate identification and ranking.</p>",
        unsafe_allow_html=True,
    )

    tab_epi, tab_metrics, tab_math, tab_filters = st.tabs([
        "Epigenomic Marks", "Ranking Metrics", "Mathematical Formulas", "Filter Criteria",
    ])

    with tab_epi:
        st.subheader("Histone Modifications")
        st.markdown("""
        Histone modifications are covalent marks on histone proteins that regulate
        chromatin accessibility and transcription. The pipeline quantifies **6 marks**
        by ChIP-seq fold-change over input across 11 cell lines
        (ENCODE / Roadmap Epigenomics consortium).
        """)
        marks_data = pd.DataFrame({
            "Mark":       ["H3K4me3", "H3K27ac", "H3K9ac", "H3K36me3", "H3K27me3", "H3K9me3"],
            "Type":       ["Active", "Active", "Active", "Active (elongation)", "Repressive", "Repressive"],
            "Deposited by": ["SET1/COMPASS", "p300/CBP", "GCN5/PCAF", "SETD2", "PRC2 (EZH2)", "SUV39H1/SETDB1"],
            "UCOE expectation": [
                "High, ubiquitous across cell types",
                "High, ubiquitous across cell types",
                "High, ubiquitous across cell types",
                "Moderate (housekeeping transcription)",
                "Absent — indicates Polycomb silencing",
                "Absent — indicates constitutive heterochromatin",
            ],
        })
        st.dataframe(marks_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("DNA Methylation (WGBS)")
        st.markdown("""
        Whole-Genome Bisulfite Sequencing (WGBS) measures 5-methylcytosine at CpG
        dinucleotides at single-base resolution.

        - **Hypomethylation** (< 10%) at CpG islands indicates constitutively active, open chromatin
        - UCOEs must remain hypomethylated across all cell types — this is the defining epigenomic property
        - The pipeline applies a strict filter: mean CpG methylation < 10% in ≥ 80% of cell lines with WGBS data

        Unmethylated CpG islands recruit the CXXC domain of CXXC1/CFP1, which in turn
        recruits the SET1/COMPASS complex to deposit H3K4me3, creating a positive
        feedback loop that sustains constitutive chromatin opening.
        """)

        st.markdown("---")
        st.subheader("DNase-seq (Chromatin Accessibility)")
        st.markdown("""
        DNase-seq identifies nucleosome-depleted regions by their hypersensitivity to
        DNase I digestion.

        - Used as a **ranking feature** in Phase II, not as a binary filter in Phase I
        - Known UCOEs show moderate, constitutive DNase signal (fold-change < 2.0)
        - High fold-change DNase signal is biased against GC-rich sequences due to GC bias
          in the input control, which would generate false negatives if used as a filter
        """)

        st.markdown("---")
        st.subheader("Repli-seq (Replication Timing)")
        st.markdown("""
        Repli-seq measures the ratio of Early-to-Late S-phase replication signal (E/L ratio).

        - **High E/L** → early replication → open, transcriptionally active chromatin
        - UCOEs are expected to replicate constitutively early, consistent with open chromatin
        """)

        st.markdown("---")
        st.subheader("CTCF Binding")
        st.markdown("""
        CTCF (CCCTC-binding factor) defines topological chromatin domain boundaries and
        can provide insulator activity that shields adjacent regions from heterochromatin spreading.

        - Quantified as the number of CTCF ChIP-seq peaks within ±2 kb of the candidate region
        - CTCF peaks flanking a UCOE may enhance transgene insulation at integrated loci
        """)

    with tab_metrics:
        st.subheader("Phase II — Multivariate Ranking")
        st.markdown("""
        Each of the 599 Phase I candidates is described by a **21-dimensional feature vector**
        and ranked by three complementary similarity metrics relative to the 3 known UCOEs.
        """)

        feat_df = pd.DataFrame({
            "Feature group":   ["H3K4me3", "H3K27ac", "H3K9ac", "H3K36me3",
                                 "H3K27me3", "H3K9me3", "Methylation", "DNase",
                                 "Repli-seq", "CTCF", "CpG Obs/Exp", "GC %", "Inter-TSS"],
            "Variables":       ["mean, CV"] * 8 + ["mean", "peak count",
                                 "ratio", "%", "bp"],
            "# features":      [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
            "Source":          ["ChIP-seq bigWig"] * 8 +
                               ["Repli-seq bigWig", "CTCF narrowPeak",
                                "UCSC CpG track", "UCSC CpG track", "GENCODE v44"],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.markdown("**Total: 21 features** (8 marks × 2 + 5 scalar features)")

        st.markdown("---")
        for title, weight, desc in [
            ("1 · Mahalanobis Distance", "Weight: 0.4",
             "Distance from each candidate to the centroid of the 3 reference UCOEs, "
             "accounting for inter-feature correlations. Covariance estimated by "
             "Ledoit-Wolf shrinkage over the 599 candidates; inverted via Moore-Penrose "
             "pseudo-inverse. Lower distance → higher score."),
            ("2 · Cosine Similarity", "Weight: 0.3",
             "Cosine of the angle between the z-score normalised feature vector of the "
             "candidate and the reference centroid. Captures profile shape independently "
             "of signal magnitude. Range: −1 to +1."),
            ("3 · Composite Percentile Rank", "Weight: 0.3",
             "For each of the 21 features, each candidate is assigned a percentile rank "
             "relative to all 599 candidates (direction adjusted per feature biology). "
             "The 21 percentiles are averaged into a single score ∈ [0, 1]."),
        ]:
            st.markdown(f"""
            <div style="background:{C_BGCARD};border:1px solid {C_BORDER};
                        border-radius:10px;padding:18px 22px;margin-bottom:12px">
            <div style="font-weight:700;color:{C_NAVY};margin-bottom:4px">{title}</div>
            <div style="font-size:0.82rem;color:{C_TEAL};font-weight:600;
                        margin-bottom:8px">{weight}</div>
            <div style="color:#5D6470;font-size:0.9rem;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with tab_math:
        st.subheader("Mathematical Formulas")

        st.markdown("#### Mahalanobis Distance")
        st.latex(r"D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu}_{ref})^\top \hat{\Sigma}^{+} (\mathbf{x} - \boldsymbol{\mu}_{ref})}")
        st.markdown("""
        - **x** ∈ ℝ²¹ — feature vector of a candidate
        - **μ_ref** = ⅓ Σᵢ xᵢ — centroid of the 3 reference UCOEs
        - **Σ̂** — Ledoit-Wolf shrinkage covariance estimated over 599 candidates
        - **Σ̂⁺** — Moore-Penrose pseudo-inverse for numerical stability
        """)

        st.markdown("---")
        st.markdown("#### Cosine Similarity")
        st.latex(r"\text{cos}(\mathbf{x},\boldsymbol{\mu}_{ref}) = \frac{\mathbf{z}_x \cdot \mathbf{z}_\mu}{\|\mathbf{z}_x\|\,\|\mathbf{z}_\mu\|}")
        st.markdown("z-score normalisation applied over all candidates + references jointly.")

        st.markdown("---")
        st.markdown("#### Composite Percentile Score")
        st.latex(r"P(\mathbf{x}) = \frac{1}{21}\sum_{j=1}^{21} p_j(\mathbf{x}), \quad p_j \in [0,1]")

        st.markdown("---")
        st.markdown("#### Composite UCOE Score")
        st.latex(r"S(\mathbf{x}) = 0.4\,\tilde{D}_M(\mathbf{x}) + 0.3\,\widetilde{\cos}(\mathbf{x}) + 0.3\,\tilde{P}(\mathbf{x})")
        st.markdown("Tilde (~) denotes min-max normalisation to [0, 1]; D_M is inverted.")

        st.markdown("---")
        st.markdown("#### Ledoit-Wolf Shrinkage Estimator")
        st.latex(r"\hat{\Sigma}_{LW} = (1-\alpha)\,\hat{\Sigma}_{sample} + \alpha\,\frac{\mathrm{tr}(\hat{\Sigma}_{sample})}{p}\,I_p")
        st.markdown("""
        - α ∈ [0, 1] — optimal shrinkage intensity estimated analytically
        - p = 21 — number of features
        - Required because n = 3 reference UCOEs yields a singular sample covariance
        """)

        st.markdown("---")
        st.markdown("#### Sensitivity Analysis")
        st.latex(r"\{(w_M, w_C, w_P) : w_M + w_C + w_P = 1,\; w_i \geq 0.05,\; w_i \in \{0.05, 0.15, \ldots\}\}")
        st.markdown("29 combinations; a candidate is *stable* if it appears in top 20 in ≥ 80% of combinations.")

    with tab_filters:
        st.subheader("Phase I Filter Criteria")

        for title, content in [
            ("Filter 1 — Bidirectional HKG Promoters (789 candidates)", """
- **Source**: GENCODE v44 protein-coding genes + Human Protein Atlas housekeeping classification
- **Criterion**: Pairs of housekeeping genes on opposite strands with TSS–TSS distance ≤ 5,000 bp
- **Patterns**: divergent classic (←→) and divergent overlapping (shared promoter)
- **Rationale**: All 3 validated UCOEs reside at bidirectional HKG promoters
            """),
            ("Filter 2 — CpG Island Overlap ≥ 40% (692 retained, 97 eliminated)", """
- **Source**: UCSC cpgIslandExt track
- **Pre-processing**: Candidate coordinates extended ±500 bp before intersection
- **Threshold**: 40% (not 50%) because TBP/PSMB1 has 46% coverage from two discontinuous CpG islands
            """),
            ("Filter 3 — Ubiquitous Active Marks (647 retained, 45 eliminated)", """
- **Marks**: H3K4me3 AND H3K27ac (both independently required)
- **Criterion**: Fold-change over input > 2.0 in ≥ 80% of available cell lines
- **Rationale**: UCOE activity must be constitutive, not tissue-restricted
            """),
            ("Filter 4 — Absence of Repressive Marks (645 retained, 2 eliminated)", """
- **Marks**: H3K27me3 (Polycomb) and H3K9me3 (constitutive heterochromatin)
- **Criterion**: Fold-change < 2.0 in ≥ 80% of cell lines
            """),
            ("Filter 5 — Constitutive Hypomethylation (599 retained, 46 eliminated)", """
- **Source**: WGBS — 8 of 11 cell lines had data
- **Criterion**: Mean CpG methylation < 10% in ≥ 80% of cell lines with WGBS data
            """),
        ]:
            with st.expander(title):
                st.markdown(content)

        st.markdown("#### Cell lines used (11 Roadmap Epigenomics)")
        cell_lines = pd.DataFrame({
            "ID":   ["E003","E004","E005","E006","E007","E011","E012","E013","E016","E024","E066"],
            "Cell line": [
                "H1 ES cells", "H1 BMP4-derived mesendoderm", "H1 BMP4-derived trophoblast",
                "H1 derived mesenchymal stem cells", "H1 derived neural progenitor",
                "hESC-derived CD184+ endoderm", "hESC-derived CD56+ ectoderm",
                "hESC-derived CD56+ mesoderm", "HUES64 ES cells",
                "ES-UCSF4 ES cells", "Liver",
            ],
        })
        st.dataframe(cell_lines, use_container_width=True, hide_index=True)
        st.caption(
            "Cell lines were selected to include pluripotent and early-differentiation "
            "states, ensuring that activity filters are not biased toward any single lineage."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Downloads":
    st.title("Downloads")
    st.markdown(
        "All datasets generated by the UCOE discovery pipeline are available below. "
        "Please cite this work if you use these data."
    )

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.subheader("Candidate Data")

        st.markdown("**Scored candidates (all 599)**")
        st.download_button(
            "Download scored_candidates.tsv",
            scored.to_csv(sep="\t", index=False),
            "ucoe_scored_candidates.tsv", "text/tab-separated-values",
        )

        st.markdown("**Integrated classification (ETS, conservation, biophysical)**")
        if integrated is not None:
            st.download_button(
                "Download ucoe_integrated_classification.tsv",
                integrated.to_csv(sep="\t", index=False),
                "ucoe_integrated_classification.tsv", "text/tab-separated-values",
            )

        st.markdown("**Top 20 candidates**")
        top20 = scored_full.nsmallest(20, "composite_rank")[[
            "composite_rank", "label", "chrom", "start", "end", "length",
            "composite_score", "mahalanobis_dist", "cosine_sim",
        ]]
        st.dataframe(top20.round(4), use_container_width=True, hide_index=True)
        st.download_button(
            "Download top20.tsv",
            top20.to_csv(sep="\t", index=False),
            "ucoe_top20.tsv", "text/tab-separated-values",
        )

    with col_dl2:
        st.subheader("Reference & Validation Data")

        st.markdown("**Reference UCOE profile (3 known UCOEs)**")
        st.download_button(
            "Download reference_profile.tsv",
            ref.to_csv(sep="\t", index=False),
            "ucoe_reference_profile.tsv", "text/tab-separated-values",
        )

        st.markdown("**LOO cross-validation results**")
        st.download_button(
            "Download loo_validation.tsv",
            loo_df.to_csv(sep="\t", index=False),
            "ucoe_loo_validation.tsv", "text/tab-separated-values",
        )

        st.markdown("**Weight sensitivity analysis**")
        st.download_button(
            "Download sensitivity_analysis.tsv",
            sens_raw.to_csv(sep="\t", index=False),
            "ucoe_sensitivity_analysis.tsv", "text/tab-separated-values",
        )

        fasta_path = DATA_DIR / "ucoe_sequences.fa"
        if fasta_path.exists():
            st.markdown("**Candidate sequences (FASTA)**")
            st.download_button(
                "Download ucoe_sequences.fa",
                fasta_path.read_text(),
                "ucoe_sequences.fa", "text/plain",
            )

    st.markdown("---")
    st.markdown(f"""
    | | |
    |---|---|
    | **Genome assembly** | GRCh38 / hg38 |
    | **Gene annotation** | GENCODE v44 |
    | **Epigenomic data** | ENCODE / Roadmap Epigenomics (11 cell lines) |
    | **Housekeeping genes** | Human Protein Atlas |
    | **CpG islands** | UCSC cpgIslandExt |
    | **Pipeline source code** | [github.com/0stetti/ucoe-discovery-pipeline](https://github.com/0stetti/ucoe-discovery-pipeline) |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About UCOE Atlas")

    st.markdown("""
    ### What are UCOEs?

    **Ubiquitous Chromatin Opening Elements (UCOEs)** are DNA regulatory sequences
    at bidirectional promoters between divergently transcribed housekeeping genes.
    They maintain constitutively open chromatin and resist epigenetic silencing of
    adjacent transgenes regardless of chromosomal integration site — a property
    exploited in stable gene expression systems for biopharmaceutical production
    and gene therapy.

    Only **3 UCOEs** have been experimentally characterised in the human genome.
    """)

    k1, k2, k3 = st.columns(3)
    for col, name, genes, chrom in [
        (k1, "A2UCOE",    "HNRNPA2B1 / CBX3", "chr12"),
        (k2, "TBP/PSMB1", "TBP / PSMB1",      "chr6"),
        (k3, "SRF-UCOE",  "SURF1 / SURF2",     "chr9"),
    ]:
        col.markdown(f"""
        <div style="background:{C_BGCARD};border:1px solid {C_BORDER};
                    border-top:3px solid {C_RED};border-radius:10px;padding:16px 18px">
        <div style="font-weight:700;color:{C_NAVY};font-size:1rem">{name}</div>
        <div style="color:{C_GRAY};font-size:0.88rem;margin:4px 0">{genes}</div>
        <div style="color:{C_BLUE};font-size:0.82rem;font-weight:600">{chrom}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    ### Pipeline overview

    **Phase I** — Sequential epigenomic filtering (789 → 599 candidates):
    1. Bidirectional HKG promoters (GENCODE v44 + Human Protein Atlas)
    2. CpG island overlap ≥ 40% (UCSC cpgIslandExt)
    3. Ubiquitous H3K4me3 + H3K27ac activity (≥ 80% of cell lines)
    4. Absence of H3K27me3 + H3K9me3 (≥ 80% of cell lines)
    5. Constitutive hypomethylation < 10% (WGBS, ≥ 80% of cell lines)

    **Phase II** — Multivariate ranking by similarity to the 3 known UCOEs,
    using Mahalanobis distance (Ledoit-Wolf covariance), cosine similarity,
    and composite percentile rank over 21 epigenomic features.

    **Validation** — Weight sensitivity analysis (29 combinations on the simplex)
    identifies 7 candidates with 100% rank stability in the top 20.

    ### Top candidate

    **SMIM27/TOPORS** (chr9:32,550,561–32,552,586 · 2,025 bp) achieves the
    highest composite score (S = 0.787), with 7 ETS motifs (3.46/kb),
    4 positioned within the nucleosome-free region, and biophysical properties
    consistent with UCOE-like chromatin architecture.

    ### Citation

    *Manuscript in preparation.*

    Ostetti, E.R.S.; Manieri, T.M.; Moro, A.M. (2026).
    Computational identification and characterisation of Ubiquitous Chromatin
    Opening Element candidates in the human genome.
    """)

    st.markdown("---")
    st.caption(
        "Elton R. Ostetti — PhD Student, University of São Paulo / Instituto Butantan  \n"
        "Tânia M. Manieri — Co-supervisor, Instituto Butantan  \n"
        "Ana M. Moro — Supervisor, Biopharmaceuticals Laboratory, Instituto Butantan"
    )
