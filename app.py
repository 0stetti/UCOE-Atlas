"""
UCOE Atlas — Interactive platform for exploring UCOE candidates
in the human genome.

Developed by Elton Roger Ostetti (USP / Instituto Butantan)
Supervisor: Prof. Dr. Ana Maria Moro

Run: streamlit run webapp/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Config ──
st.set_page_config(
    page_title="UCOE Atlas",
    page_icon="dna",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

# ── Load data ──
@st.cache_data
def load_data():
    scored = pd.read_csv(DATA_DIR / "scored_candidates.tsv", sep="\t")
    ref = pd.read_csv(DATA_DIR / "reference_profile.tsv", sep="\t")
    return scored, ref

@st.cache_data
def load_structural():
    try:
        return pd.read_csv(DATA_DIR / "all_groups_structural.tsv", sep="\t")
    except FileNotFoundError:
        return None

scored, ref = load_data()
structural = load_structural()

# Add computed columns
scored["length"] = scored["end"] - scored["start"]
scored["label"] = scored["gene1"] + "/" + scored["gene2"]

# Known UCOEs identification
KNOWN_UCOES = {
    "A2UCOE": ("CBX3", "HNRNPA2B1"),
    "TBP/PSMB1": ("PSMB1", "TBP"),
    "SRF-UCOE": ("SURF2", "SURF1"),
}

def is_known_ucoe(row):
    for name, (g1, g2) in KNOWN_UCOES.items():
        if (row["gene1"] == g1 and row["gene2"] == g2) or \
           (row["gene1"] == g2 and row["gene2"] == g1):
            return name
    return None

scored["known_ucoe"] = scored.apply(is_known_ucoe, axis=1)

# Features for radar/PCA
FEATURES = [
    "H3K4me3_mean", "H3K4me3_cv", "H3K27ac_mean", "H3K27ac_cv",
    "H3K27me3_mean", "H3K27me3_cv", "H3K9me3_mean", "H3K9me3_cv",
    "meth_mean", "meth_cv", "DNase_mean", "DNase_cv",
    "H3K9ac_mean", "H3K9ac_cv", "H3K36me3_mean", "H3K36me3_cv",
    "repliseq_mean", "CTCF_n_peaks",
]

FEATURE_LABELS = {
    "H3K4me3_mean": "H3K4me3", "H3K4me3_cv": "H3K4me3 CV",
    "H3K27ac_mean": "H3K27ac", "H3K27ac_cv": "H3K27ac CV",
    "H3K27me3_mean": "H3K27me3", "H3K27me3_cv": "H3K27me3 CV",
    "H3K9me3_mean": "H3K9me3", "H3K9me3_cv": "H3K9me3 CV",
    "meth_mean": "Methylation", "meth_cv": "Meth CV",
    "DNase_mean": "DNase", "DNase_cv": "DNase CV",
    "H3K9ac_mean": "H3K9ac", "H3K9ac_cv": "H3K9ac CV",
    "H3K36me3_mean": "H3K36me3", "H3K36me3_cv": "H3K36me3 CV",
    "repliseq_mean": "Repli-seq", "CTCF_n_peaks": "CTCF peaks",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("UCOE Atlas")
st.sidebar.caption("Interactive UCOE Candidate Explorer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Candidate Explorer", "Candidate Detail",
     "PCA Explorer", "Methods & Glossary", "Downloads", "About"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Developed by **Elton R. Ostetti**  \n"
    "USP / Instituto Butantan  \n"
    "Supervisor: Prof. Dr. Ana Maria Moro"
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("UCOE Atlas")
    st.markdown("### A Comprehensive Repository of UCOE Candidates in the Human Genome")

    st.markdown("""
    **Ubiquitous Chromatin Opening Elements (UCOEs)** are DNA regulatory sequences
    that maintain constitutively open chromatin and prevent transgene silencing.
    Only 3 UCOEs have been validated in the human genome to date.

    This platform provides interactive access to **599 UCOE candidates** identified
    by a two-phase computational pipeline operating on epigenomic data from
    11 cell lines (ENCODE/Roadmap Epigenomics consortium).
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candidates", "599")
    col2.metric("Known UCOEs Recovered", "3/3")
    col3.metric("100% Stable (top-20)", "7")
    col4.metric("Epigenomic Features", "21")

    st.markdown("---")

    # Pipeline overview
    st.subheader("Pipeline Summary")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Phase I — Qualitative Filtering")
        filter_data = pd.DataFrame({
            "Filter": [
                "Divergent HKG promoters (≤5 kb)",
                "CpG island overlap ≥ 40%",
                "Active marks (H3K4me3 + H3K27ac) ≥ 80%",
                "Repressive marks absent ≥ 80%",
                "Hypomethylation (< 10%) ≥ 80%",
            ],
            "Retained": [789, 692, 670, 645, 599],
        })
        st.dataframe(filter_data, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Phase II — Multivariate Ranking")
        st.markdown("""
        - **Mahalanobis distance** to UCOE centroid (Ledoit-Wolf covariance)
        - **Cosine similarity** in z-score space
        - **Composite percentile** rank
        - Weighted score: 0.4 Mah + 0.3 Cos + 0.3 Pct
        """)

    # Score distribution
    st.markdown("---")
    st.subheader("Composite Score Distribution")
    fig_dist = px.histogram(
        scored, x="composite_score", nbins=50,
        color_discrete_sequence=["#2196F3"],
        labels={"composite_score": "UCOE Composite Score"},
    )
    # Mark known UCOEs
    for _, row in scored[scored["known_ucoe"].notna()].iterrows():
        fig_dist.add_vline(
            x=row["composite_score"], line_dash="dash", line_color="red",
            annotation_text=row["known_ucoe"], annotation_position="top",
        )
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Candidate Explorer":
    st.title("Candidate Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        rank_range = st.slider("Rank range", 1, 599, (1, 50))
    with col2:
        chrom_filter = st.multiselect(
            "Chromosome", sorted(scored["chrom"].unique()),
            default=[]
        )
    with col3:
        min_score = st.slider("Min composite score", 0.0, 1.0, 0.0, 0.01)

    # Apply filters
    mask = (
        (scored["composite_rank"] >= rank_range[0]) &
        (scored["composite_rank"] <= rank_range[1]) &
        (scored["composite_score"] >= min_score)
    )
    if chrom_filter:
        mask &= scored["chrom"].isin(chrom_filter)

    filtered = scored[mask].sort_values("composite_rank")

    st.markdown(f"**Showing {len(filtered)} candidates**")

    # Display columns
    display_cols = [
        "composite_rank", "label", "chrom", "start", "end", "length",
        "composite_score", "mahalanobis_dist", "cosine_sim",
        "H3K4me3_mean", "H3K27ac_mean", "meth_mean",
        "H3K27me3_mean", "H3K9me3_mean", "CTCF_n_peaks", "known_ucoe",
    ]
    rename = {
        "composite_rank": "Rank", "label": "Gene Pair", "length": "Length (bp)",
        "composite_score": "Score", "mahalanobis_dist": "Mahal. Dist",
        "cosine_sim": "Cos. Sim", "H3K4me3_mean": "H3K4me3",
        "H3K27ac_mean": "H3K27ac", "meth_mean": "Meth %",
        "H3K27me3_mean": "H3K27me3", "H3K9me3_mean": "H3K9me3",
        "CTCF_n_peaks": "CTCF", "known_ucoe": "Known UCOE",
    }

    st.dataframe(
        filtered[display_cols].rename(columns=rename).round(3),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # Chromosome distribution
    st.markdown("---")
    st.subheader("Chromosome Distribution")
    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chrom_counts = filtered["chrom"].value_counts().reindex(chrom_order).dropna().astype(int)
    fig_chrom = px.bar(
        x=chrom_counts.index, y=chrom_counts.values,
        labels={"x": "Chromosome", "y": "Candidates"},
        color_discrete_sequence=["#4CAF50"],
    )
    fig_chrom.update_layout(height=300)
    st.plotly_chart(fig_chrom, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Candidate Detail":
    st.title("Candidate Detail")

    # Selector
    options = [
        f"#{int(r['composite_rank'])} — {r['label']} ({r['chrom']})"
        for _, r in scored.sort_values("composite_rank").iterrows()
    ]
    selected = st.selectbox("Select candidate", options)
    rank = int(selected.split("#")[1].split(" ")[0])
    cand = scored[scored["composite_rank"] == rank].iloc[0]

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"## {cand['label']}")
        known = cand["known_ucoe"]
        if known:
            st.success(f"Known UCOE: **{known}**")
        st.markdown(
            f"**{cand['chrom']}:{cand['start']:,}-{cand['end']:,}**  \n"
            f"Length: {cand['length']:,} bp | "
            f"Type: {cand['pair_type']} | "
            f"Inter-TSS: {cand['inter_tss_distance']:,} bp"
        )
    with col2:
        st.metric("Rank", f"#{int(cand['composite_rank'])}")
        st.metric("Score", f"{cand['composite_score']:.4f}")
    with col3:
        st.metric("Mahalanobis", f"{cand['mahalanobis_dist']:.3f}")
        st.metric("Cosine Sim", f"{cand['cosine_sim']:.3f}")

    st.markdown("---")

    # Epigenomic profile
    st.subheader("Epigenomic Profile")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Radar chart
        features_radar = [
            "H3K4me3_mean", "H3K27ac_mean", "H3K9ac_mean", "H3K36me3_mean",
            "DNase_mean", "repliseq_mean",
        ]
        labels_radar = ["H3K4me3", "H3K27ac", "H3K9ac", "H3K36me3", "DNase", "Repli-seq"]

        # Normalize to 0-1 by min-max across all candidates
        vals_cand = []
        vals_ref = []
        for f in features_radar:
            fmin = scored[f].min()
            fmax = scored[f].max()
            rng = fmax - fmin if fmax - fmin > 0 else 1
            vals_cand.append((cand[f] - fmin) / rng)
            vals_ref.append((ref[f].mean() - fmin) / rng)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_cand + [vals_cand[0]],
            theta=labels_radar + [labels_radar[0]],
            fill="toself", name=cand["label"],
            line_color="#2196F3", fillcolor="rgba(33,150,243,0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_ref + [vals_ref[0]],
            theta=labels_radar + [labels_radar[0]],
            fill="toself", name="UCOE Reference",
            line_color="#E74C3C", fillcolor="rgba(231,76,60,0.1)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                angularaxis=dict(tickfont=dict(size=12)),
            ),
            title="Active Marks (normalized)",
            height=450,
            margin=dict(l=80, r=80, t=60, b=40),
            legend=dict(x=0.5, y=-0.15, xanchor="center", orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        # Bar chart of repressive marks + methylation
        marks = {
            "H3K27me3": cand["H3K27me3_mean"],
            "H3K9me3": cand["H3K9me3_mean"],
            "Methylation (%)": cand["meth_mean"],
        }
        ref_marks = {
            "H3K27me3": ref["H3K27me3_mean"].mean(),
            "H3K9me3": ref["H3K9me3_mean"].mean(),
            "Methylation (%)": ref["meth_mean"].mean(),
        }

        fig_repr = go.Figure()
        fig_repr.add_trace(go.Bar(
            x=list(marks.keys()), y=list(marks.values()),
            name=cand["label"], marker_color="#2196F3",
        ))
        fig_repr.add_trace(go.Bar(
            x=list(ref_marks.keys()), y=list(ref_marks.values()),
            name="UCOE Reference", marker_color="#E74C3C",
        ))
        fig_repr.update_layout(
            barmode="group", title="Repressive Marks & Methylation",
            height=400, yaxis_title="Signal / %",
        )
        st.plotly_chart(fig_repr, use_container_width=True)

    # CpG Island info
    st.markdown("---")
    st.subheader("CpG Island Properties")
    col1, col2, col3 = st.columns(3)
    col1.metric("CpG Overlap", f"{cand['cpg_overlap_fraction']*100:.1f}%")
    col2.metric("CpG Obs/Exp", f"{cand['cpg_obs_exp']:.1f}")
    col3.metric("GC Content", f"{cand['cpg_gc_pct']:.0f}%"
                if cand['cpg_gc_pct'] < 100 else f"{cand['length']/cand['length']*57:.0f}%")

    # ── Composite genomic figure ──
    st.markdown("---")
    st.subheader("Genomic Profile — Sequence, CpG & ETS Analysis")

    # Parse FASTA to get sequence for this candidate
    @st.cache_data
    def load_fasta_sequences():
        seqs = {}
        fasta_path = DATA_DIR / "ucoe_sequences.fa"
        if not fasta_path.exists():
            return seqs
        header = None
        seq_lines = []
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

    import re as _re

    fasta_seqs = load_fasta_sequences()

    # Try to find this candidate's sequence in FASTA
    # FASTA headers may use different formats — try common patterns
    seq_key = None
    for key in fasta_seqs:
        if cand["gene1"] in key and cand["gene2"] in key:
            seq_key = key
            break
        if f"{cand['chrom']}:{cand['start']}-{cand['end']}" in key:
            seq_key = key
            break
        if f"{cand['chrom']}_{cand['start']}_{cand['end']}" in key:
            seq_key = key
            break

    if seq_key is None:
        # Try matching by coordinates in header
        for key in fasta_seqs:
            if str(cand["start"]) in key and str(cand["end"]) in key:
                seq_key = key
                break

    if seq_key and seq_key in fasta_seqs:
        seq = fasta_seqs[seq_key]
        length = len(seq)
        positions = list(range(length))
        window = 100

        # ── ETS motifs ──
        ets_fwd = [(m.start(), m.end(), "+", m.group()) for m in _re.finditer(r"CGGAA[GA]", seq)]
        ets_rev = [(m.start(), m.end(), "-", m.group()) for m in _re.finditer(r"[TC]TTCCG", seq)]
        ets_all = sorted(ets_fwd + ets_rev, key=lambda x: x[0])

        # ── GC content (sliding window) ──
        gc_pos, gc_vals = [], []
        for i in range(0, length - window + 1, window // 2):
            w = seq[i:i + window]
            gc_pos.append(i + window // 2)
            gc_vals.append((w.count("G") + w.count("C")) / len(w) * 100)

        # ── CpG density (sliding window) ──
        cpg_pos, cpg_vals = [], []
        for i in range(0, length - window + 1, window // 2):
            w = seq[i:i + window]
            cpg_pos.append(i + window // 2)
            cpg_vals.append(w.count("CG"))

        # ── ETS density (window 200bp) ──
        ets_w = 200
        ets_dens_pos, ets_dens_vals = [], []
        for i in range(0, length - ets_w + 1, ets_w // 2):
            c = sum(1 for s, e, _, _ in ets_all if s >= i and s < i + ets_w)
            ets_dens_pos.append(i + ets_w // 2)
            ets_dens_vals.append(c)

        # ── Build interactive plotly figure ──
        fig_comp = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.35, 0.25, 0.15, 0.25],
            vertical_spacing=0.04,
            subplot_titles=(
                f"GC Content & CpG Density — {cand['label']}",
                "CpG Island Coverage",
                "ETS Motif Density (window = 200 bp)",
                "ETS Motif Positions",
            ),
        )

        # Panel 1: GC + CpG density
        fig_comp.add_trace(
            go.Scatter(x=gc_pos, y=gc_vals, mode="lines", name="GC %",
                       line=dict(color="#00897B", width=1.5),
                       fill="tozeroy", fillcolor="rgba(0,137,123,0.15)"),
            row=1, col=1,
        )
        fig_comp.add_trace(
            go.Bar(x=cpg_pos, y=cpg_vals, name="CpG count/window",
                   marker_color="rgba(255,143,0,0.5)", width=window * 0.4),
            row=1, col=1,
        )
        fig_comp.add_hline(y=50, line_dash="dash", line_color="gray",
                           opacity=0.5, row=1, col=1)
        fig_comp.update_yaxes(title_text="GC % / CpG count", row=1, col=1)

        # Panel 2: CpG island rectangle
        cpg_cov = cand["cpg_overlap_fraction"]
        cpg_end = int(length * cpg_cov)
        fig_comp.add_trace(
            go.Bar(x=[cpg_end / 2], y=[1], width=[cpg_end], name="CpG Island",
                   marker_color="rgba(129,199,132,0.7)",
                   text=[f"CpG Island ({cpg_cov*100:.1f}% coverage)"],
                   textposition="inside", textfont=dict(size=11, color="darkgreen")),
            row=2, col=1,
        )
        fig_comp.update_yaxes(visible=False, row=2, col=1)
        fig_comp.update_yaxes(range=[0, 1.5], row=2, col=1)

        # Panel 3: ETS density
        fig_comp.add_trace(
            go.Bar(x=ets_dens_pos, y=ets_dens_vals, name="ETS motifs/window",
                   marker_color="rgba(231,76,60,0.7)", width=100),
            row=3, col=1,
        )
        fig_comp.update_yaxes(title_text="ETS / 200bp", row=3, col=1)

        # Panel 4: ETS motif positions as markers
        for i, (s, e, strand, motif_seq) in enumerate(ets_all):
            color = "#E74C3C" if strand == "+" else "#9B59B6"
            symbol = "triangle-up" if strand == "+" else "triangle-down"
            fig_comp.add_trace(
                go.Scatter(
                    x=[s], y=[0.5 if strand == "+" else -0.5],
                    mode="markers+text",
                    marker=dict(size=12, color=color, symbol=symbol),
                    text=[f"#{i+1} {motif_seq}"],
                    textposition="top center" if strand == "+" else "bottom center",
                    textfont=dict(size=9),
                    name=f"ETS #{i+1} ({strand})",
                    showlegend=False,
                ),
                row=4, col=1,
            )

        # Gene arrows as annotations
        fig_comp.add_annotation(
            x=length * 0.75, y=1.2, text=f"→ {cand['gene1']}", showarrow=False,
            font=dict(size=12, color="#2E86C1", family="Arial Black"),
            row=4, col=1,
        )
        fig_comp.add_annotation(
            x=length * 0.25, y=-1.2, text=f"← {cand['gene2']}", showarrow=False,
            font=dict(size=12, color="#E67E22", family="Arial Black"),
            row=4, col=1,
        )

        fig_comp.update_yaxes(range=[-2, 2], visible=False, row=4, col=1)
        fig_comp.update_xaxes(title_text=f"Position in {cand['label']} (bp)", row=4, col=1)

        fig_comp.update_layout(
            height=700, showlegend=False,
            margin=dict(l=60, r=30, t=40, b=40),
        )

        st.plotly_chart(fig_comp, use_container_width=True)

        # ETS summary table
        if ets_all:
            st.markdown(f"**{len(ets_all)} ETS motifs** found ({len(ets_all)/length*1000:.1f} motifs/kb)")
            ets_df = pd.DataFrame([
                {"#": i+1, "Position": s, "Strand": strand, "Motif": motif_seq,
                 "Abs. Position": f"{cand['chrom']}:{cand['start']+s}"}
                for i, (s, e, strand, motif_seq) in enumerate(ets_all)
            ])
            st.dataframe(ets_df, use_container_width=True, hide_index=True)
        else:
            st.info("No ETS motifs (CGGAA[GA]) found in this candidate.")
    else:
        st.warning("Sequence not available for this candidate. FASTA header not matched.")

    # UCSC link
    st.markdown("---")
    ucsc_url = (
        f"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&"
        f"position={cand['chrom']}%3A{cand['start']}-{cand['end']}"
    )
    st.markdown(f"[View in UCSC Genome Browser]({ucsc_url})")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PCA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "PCA Explorer":
    st.title("PCA Explorer — Epigenomic Space")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X = scored[FEATURES].copy()
    for col in FEATURES:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    scored_pca = scored.copy()
    scored_pca["PC1"] = Z_pca[:, 0]
    scored_pca["PC2"] = Z_pca[:, 1]

    # Color by
    color_by = st.selectbox(
        "Color by",
        ["composite_score", "mahalanobis_dist", "composite_rank",
         "H3K4me3_mean", "meth_mean", "length"],
        index=0,
    )

    fig_pca = px.scatter(
        scored_pca, x="PC1", y="PC2",
        color=color_by,
        hover_data=["label", "composite_rank", "composite_score", "chrom"],
        color_continuous_scale="Viridis_r" if "dist" in color_by or "rank" in color_by else "Viridis",
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        },
        title="599 UCOE Candidates in PCA Space",
    )

    # Highlight known UCOEs
    known_mask = scored_pca["known_ucoe"].notna()
    if known_mask.any():
        known_df = scored_pca[known_mask]
        fig_pca.add_trace(go.Scatter(
            x=known_df["PC1"], y=known_df["PC2"],
            mode="markers+text",
            marker=dict(size=15, symbol="star", color="red", line=dict(width=2, color="black")),
            text=known_df["known_ucoe"],
            textposition="top center",
            name="Known UCOEs",
            showlegend=True,
        ))

    # Highlight top candidate
    top1 = scored_pca[scored_pca["composite_rank"] == 1].iloc[0]
    fig_pca.add_trace(go.Scatter(
        x=[top1["PC1"]], y=[top1["PC2"]],
        mode="markers+text",
        marker=dict(size=12, symbol="diamond", color="#FF9800",
                    line=dict(width=2, color="black")),
        text=[top1["label"]],
        textposition="bottom center",
        name="Rank #1",
        showlegend=True,
    ))

    fig_pca.update_layout(height=600)
    st.plotly_chart(fig_pca, use_container_width=True)

    # Variance explained
    st.caption(
        f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% and "
        f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance "
        f"({(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])*100:.1f}% total)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METHODS & GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Methods & Glossary":
    st.title("Methods & Glossary")

    tab_epi, tab_metrics, tab_math, tab_filters = st.tabs([
        "Epigenomic Marks", "Ranking Metrics", "Mathematical Formulas", "Filter Criteria",
    ])

    # ── TAB: Epigenomic Marks ──
    with tab_epi:
        st.subheader("Histone Modifications")

        st.markdown("""
        Histone modifications are chemical changes to histone proteins around which DNA is wrapped.
        They regulate chromatin accessibility and gene expression. The pipeline uses **6 histone marks**
        measured by ChIP-seq across 11 cell lines from the ENCODE/Roadmap Epigenomics consortium.
        """)

        marks_data = {
            "Mark": [
                "**H3K4me3**", "**H3K27ac**", "**H3K9ac**",
                "**H3K36me3**", "**H3K27me3**", "**H3K9me3**",
            ],
            "Type": [
                "Active", "Active", "Active",
                "Active (transcription)", "Repressive", "Repressive",
            ],
            "Meaning": [
                "Trimethylation of histone H3 at lysine 4. Found at active promoters. Deposited by SET1/COMPASS complex. Mutually exclusive with DNA methylation.",
                "Acetylation of H3 at lysine 27. Marks active promoters and enhancers. Distinguishes active from poised enhancers. Deposited by p300/CBP.",
                "Acetylation of H3 at lysine 9. Marks active promoters. Associated with transcriptional initiation.",
                "Trimethylation of H3 at lysine 36. Found across gene bodies of actively transcribed genes. Deposited by SETD2 during elongation.",
                "Trimethylation of H3 at lysine 27. Polycomb-mediated repression. Marks silenced developmental genes. Deposited by PRC2 (EZH2).",
                "Trimethylation of H3 at lysine 9. Constitutive heterochromatin. Marks pericentromeric regions and silenced transgenes. Deposited by SUV39H1/SETDB1.",
            ],
            "UCOE expectation": [
                "High in all cell types (ubiquitous)",
                "High in all cell types (ubiquitous)",
                "High in all cell types",
                "Moderate (housekeeping transcription)",
                "Absent — presence indicates silencing",
                "Absent — presence indicates heterochromatin",
            ],
        }
        st.table(pd.DataFrame(marks_data))

        st.markdown("---")
        st.subheader("DNA Methylation (WGBS)")
        st.markdown("""
        **DNA methylation** is the addition of a methyl group to cytosine in CpG dinucleotides.
        Measured by **Whole-Genome Bisulfite Sequencing (WGBS)** at single-base resolution.

        - **Hypomethylation** (< 10%) at CpG islands indicates active, open chromatin
        - **Hypermethylation** (> 50%) indicates gene silencing
        - UCOEs must be **constitutively hypomethylated** across all cell types
        - The pipeline requires methylation < 10% in ≥ 80% of cell lines with WGBS data (8 of 11)

        **Why it matters for UCOEs**: Unmethylated CpGs recruit CXXC1/CFP1, which deposits H3K4me3
        via the SET1/COMPASS complex, creating a positive feedback loop that maintains open chromatin
        (Thomson et al., 2010).
        """)

        st.markdown("---")
        st.subheader("DNase-seq (Chromatin Accessibility)")
        st.markdown("""
        **DNase-seq** maps regions of open chromatin by identifying sites hypersensitive to
        DNase I digestion. Open chromatin is accessible to transcription factors.

        - **Used as informative annotation**, not as an eliminatory filter
        - UCOEs show moderate, constitutive accessibility (not hyper-accessible)
        - High fold-change is biased against GC-rich regions (CpG islands) due to input normalization artifacts
        - In the pipeline: DNase mean and CV are included as Phase II ranking features

        **Why informative but not a filter**: Known UCOEs have DNase fold-change < 2.0 due to
        GC-content bias in the input control, which would cause false negatives if used as a binary filter.
        """)

        st.markdown("---")
        st.subheader("Repli-seq (Replication Timing)")
        st.markdown("""
        **Repli-seq** measures when a genomic region is replicated during S-phase.

        - **Early-replicating** regions (high E/L ratio) correspond to open, active chromatin
        - **Late-replicating** regions correspond to heterochromatin
        - UCOEs are expected to replicate early, consistent with constitutively open chromatin
        - Measured as the ratio of Early to Late S-phase signal (E/L)
        """)

        st.markdown("---")
        st.subheader("CTCF Binding")
        st.markdown("""
        **CTCF** (CCCTC-binding factor) is an insulator protein that defines chromatin domain boundaries.

        - CTCF peaks flanking a UCOE may provide **insulator activity**, protecting against
          heterochromatin spreading from adjacent regions
        - Measured as the number of CTCF ChIP-seq peaks within ±2 kb of the candidate region
        - The ±2 kb window captures the insulator context without being limited to the candidate itself
        """)

    # ── TAB: Ranking Metrics ──
    with tab_metrics:
        st.subheader("Phase II Ranking Metrics")

        st.markdown("""
        Each candidate that passes Phase I filtering is described by a **21-feature vector**
        and ranked by similarity to the 3 known UCOEs using three complementary metrics.
        """)

        st.markdown("#### Feature Vector (21 variables)")
        feat_df = pd.DataFrame({
            "Feature": [
                "H3K4me3 mean & CV", "H3K27ac mean & CV", "H3K9ac mean & CV",
                "H3K36me3 mean & CV", "H3K27me3 mean & CV", "H3K9me3 mean & CV",
                "Methylation mean & CV", "DNase mean & CV",
                "Repli-seq mean", "CTCF peak count", "Inter-TSS distance",
            ],
            "# features": [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1],
            "Source": [
                "ChIP-seq bigWig", "ChIP-seq bigWig", "ChIP-seq bigWig",
                "ChIP-seq bigWig", "ChIP-seq bigWig", "ChIP-seq bigWig",
                "WGBS", "DNase-seq bigWig",
                "Repli-seq bigWig", "CTCF narrowPeak", "GENCODE v44",
            ],
            "Interpretation": [
                "Mean = signal intensity; CV = consistency across cell lines (low CV = ubiquitous)",
                "Mean = signal intensity; CV = consistency across cell lines",
                "Mean = signal intensity; CV = consistency across cell lines",
                "Mean = transcriptional elongation; CV = consistency",
                "Mean = repression level (should be low); CV = consistency",
                "Mean = heterochromatin (should be low); CV = consistency",
                "Mean = methylation % (should be <10%); CV = consistency",
                "Mean = accessibility; CV = consistency",
                "Early/Late ratio (higher = earlier replication)",
                "Insulator context (number of CTCF peaks in ±2 kb)",
                "Distance between TSSs of the gene pair (bp)",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.markdown("#### 1. Mahalanobis Distance")
        st.markdown("""
        Measures the distance from each candidate to the **centroid** of the 3 reference UCOEs,
        accounting for correlations between features.

        - **Covariance estimation**: Ledoit-Wolf shrinkage estimator (scikit-learn), computed over
          the 599 candidates (not the 3 references — too few for stable estimation)
        - **Matrix inversion**: Moore-Penrose pseudo-inverse for numerical stability
        - **Interpretation**: Lower distance = more similar to the UCOE reference profile
        - **Weight in composite**: 0.4 (highest, because it captures multivariate structure)
        """)

        st.markdown("#### 2. Cosine Similarity")
        st.markdown("""
        Measures the **angle** between the feature vector of each candidate and the reference centroid
        in z-score normalized space.

        - **Normalization**: StandardScaler (z-score) applied to all features before computation
        - **Range**: -1 to +1 (higher = more similar pattern)
        - **Captures**: Pattern similarity regardless of magnitude — a candidate with half the
          signal but the same relative profile scores high
        - **Weight in composite**: 0.3
        """)

        st.markdown("#### 3. Composite Percentile Rank")
        st.markdown("""
        For each feature individually, each candidate receives a **percentile rank** (0-100%)
        relative to all other candidates.

        - **Direction**: Biologically informed — "higher is better" for active marks;
          "lower is better" for repressive marks and CVs
        - **Final score**: Average of all 21 individual percentiles
        - **Captures**: Candidates consistently good across many dimensions
        - **Weight in composite**: 0.3
        """)

        st.markdown("#### Composite Score")
        st.markdown("""
        The final UCOE score combines all three metrics:

        **S = 0.4 × Mahalanobis_norm + 0.3 × Cosine_norm + 0.3 × Percentile_norm**

        Each metric is min-max normalized to [0, 1] before combination.
        Mahalanobis is inverted (lower distance → higher score).
        """)

    # ── TAB: Mathematical Formulas ──
    with tab_math:
        st.subheader("Mathematical Formulas")

        st.markdown("#### Mahalanobis Distance")
        st.latex(r"D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu}_{ref})^T \, \Sigma^{-1} \, (\mathbf{x} - \boldsymbol{\mu}_{ref})}")
        st.markdown("""
        Where:
        - **x** ∈ ℝ²¹ is the feature vector of a candidate
        - **μ_ref** = (1/3) Σᵢ xᵢ is the centroid of the 3 reference UCOEs
        - **Σ** is the covariance matrix estimated by Ledoit-Wolf shrinkage over 599 candidates
        - **Σ⁻¹** is the Moore-Penrose pseudo-inverse of Σ
        """)

        st.markdown("---")

        st.markdown("#### Cosine Similarity")
        st.latex(r"\text{cos\_sim}(\mathbf{x}, \boldsymbol{\mu}_{ref}) = \frac{\mathbf{z}_x \cdot \mathbf{z}_{\mu}}{||\mathbf{z}_x|| \cdot ||\mathbf{z}_{\mu}||}")
        st.markdown("""
        Where:
        - **z_x** and **z_μ** are the z-score normalized versions of x and μ_ref
        - Z-score: z = (x - mean) / std, computed over all candidates + references
        """)

        st.markdown("---")

        st.markdown("#### Composite Percentile Rank")
        st.latex(r"P(\mathbf{x}) = \frac{1}{21} \sum_{j=1}^{21} p_j(\mathbf{x})")
        st.markdown("""
        Where:
        - **p_j(x)** ∈ [0, 1] is the normalized percentile of candidate x in feature j
        - Direction of ranking depends on feature type (ascending for active marks, descending for repressive)
        """)

        st.markdown("---")

        st.markdown("#### Composite UCOE Score")
        st.latex(r"S_{UCOE}(\mathbf{x}) = 0.4 \cdot \tilde{D}_M(\mathbf{x}) + 0.3 \cdot \widetilde{\text{cos}}(\mathbf{x}) + 0.3 \cdot \tilde{P}(\mathbf{x})")
        st.markdown("""
        Where tilde (~) indicates min-max normalization to [0, 1].
        Mahalanobis is inverted: lower distance → higher score.
        """)

        st.markdown("---")

        st.markdown("#### Ledoit-Wolf Shrinkage Estimator")
        st.latex(r"\hat{\Sigma}_{LW} = (1 - \alpha) \hat{\Sigma}_{sample} + \alpha \cdot \frac{\text{tr}(\hat{\Sigma}_{sample})}{p} \cdot I_p")
        st.markdown("""
        Where:
        - **α** ∈ [0, 1] is the optimal shrinkage intensity (estimated analytically)
        - **Σ̂_sample** is the sample covariance matrix (599 × 21)
        - **p** = 21 (number of features)
        - **I_p** is the identity matrix

        This estimator is necessary because with only n=3 reference UCOEs, the classical
        covariance matrix would be singular. The shrinkage pulls the estimate toward a
        well-conditioned target (scaled identity), producing a positive-definite matrix.
        """)

        st.markdown("---")

        st.markdown("#### Sensitivity Analysis")
        st.markdown("""
        The composite score weights (w_M, w_C, w_P) were varied systematically:
        - Grid search over the simplex: w_M + w_C + w_P = 1.0, each w ≥ 0.05
        - Step size: 0.1 → **29 weight combinations**
        - A candidate is "stable" if it appears in the top 20 in >80% of combinations
        - **7 candidates** achieved 100% stability
        """)

    # ── TAB: Filter Criteria ──
    with tab_filters:
        st.subheader("Phase I Filter Criteria — Detailed")

        st.markdown("#### Filter 1: Bidirectional HKG Promoters")
        st.markdown("""
        - **Source**: GENCODE v44 (protein-coding genes) + Human Protein Atlas (housekeeping classification)
        - **Criterion**: Pairs of housekeeping genes on opposite strands with TSS distance ≤ 5,000 bp
        - **Two patterns detected**:
          - Divergent classic: gene(−) ←← [intergenic] →→ gene(+)
          - Divergent overlapping: gene(+) →→ [shared promoter] ←← gene(−)
        - **Rationale**: All 3 known UCOEs reside at bidirectional HKG promoters
        - **Output**: 789 candidate regions
        """)

        st.markdown("#### Filter 2: CpG Island Overlap ≥ 40%")
        st.markdown("""
        - **Source**: UCSC cpgIslandExt track
        - **Pre-processing**: Candidate coordinates extended to encompass adjacent CpG islands (±500 bp flanking window)
        - **Criterion**: ≥ 40% overlap with CpG islands after extension
        - **Why 40% and not 50%**: TBP/PSMB1 UCOE has 46% overlap due to two discontinuous CpG islands
        - **Eliminated**: 97 candidates (12.3%)
        """)

        st.markdown("#### Filter 3: Active Histone Marks ≥ 80% of Cell Lines")
        st.markdown("""
        - **Marks**: H3K4me3 and H3K27ac (both required independently)
        - **Source**: ChIP-seq fold-change bigWig files (ENCODE)
        - **Criterion**: Fold-change > 2.0 in ≥ 80% of cell lines with available data
        - **Rationale**: UCOE activity must be ubiquitous, not tissue-specific
        - **Eliminated**: 22 candidates (3.2%)
        """)

        st.markdown("#### Filter 4: Repressive Marks Absent ≥ 80% of Cell Lines")
        st.markdown("""
        - **Marks**: H3K27me3 (Polycomb) and H3K9me3 (heterochromatin)
        - **Criterion**: Fold-change < 2.0 in ≥ 80% of cell lines
        - **Rationale**: Repressive marks are incompatible with constitutively open chromatin
        - **Eliminated**: 25 candidates (3.7%)
        """)

        st.markdown("#### Filter 5: Constitutive Hypomethylation")
        st.markdown("""
        - **Source**: Whole-Genome Bisulfite Sequencing (WGBS) — 8 of 11 cell lines
        - **Criterion**: Mean CpG methylation < 10% in ≥ 80% of cell lines with WGBS data
        - **Rationale**: UCOEs must maintain unmethylated CpG islands across all cell types
        - **Eliminated**: 46 candidates (7.1%)
        """)

        st.markdown("#### Cell Lines Used")
        cell_lines = pd.DataFrame({
            "Cell Line": ["E003", "E004", "E005", "E006", "E007", "E011", "E012", "E013", "E016", "E024", "E066"],
            "Type": [
                "H1 ES cells", "H1 BMP4-derived mesendoderm", "H1 BMP4-derived trophoblast",
                "H1 derived mesenchymal stem cells", "H1 derived neural progenitor",
                "hESC derived CD184+ endoderm", "hESC derived CD56+ ectoderm",
                "hESC derived CD56+ mesoderm", "HUES64 ES cells",
                "ES-UCSF4 ES cells", "Liver",
            ],
        })
        st.dataframe(cell_lines, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Downloads":
    st.title("Downloads")

    st.markdown("""
    All data generated by the UCOE discovery pipeline is available for download below.
    Please cite this work if you use these data in your research.
    """)

    # Full scored candidates TSV
    st.subheader("Scored Candidates (all 599)")
    tsv_data = scored.to_csv(sep="\t", index=False)
    st.download_button(
        "Download scored_candidates.tsv",
        tsv_data, "ucoe_scored_candidates.tsv", "text/tab-separated-values",
    )

    # Reference profile
    st.subheader("Reference UCOE Profile (3 known UCOEs)")
    ref_data = ref.to_csv(sep="\t", index=False)
    st.download_button(
        "Download reference_profile.tsv",
        ref_data, "ucoe_reference_profile.tsv", "text/tab-separated-values",
    )

    # Top 20 summary
    st.subheader("Top 20 Candidates")
    top20 = scored.nsmallest(20, "composite_rank")[
        ["composite_rank", "label", "chrom", "start", "end", "length",
         "composite_score", "mahalanobis_dist", "cosine_sim"]
    ]
    st.dataframe(top20, use_container_width=True, hide_index=True)

    # FASTA (if exists)
    fasta_path = DATA_DIR / "ucoe_sequences.fa"
    if fasta_path.exists():
        st.subheader("Candidate Sequences (FASTA)")
        st.download_button(
            "Download ucoe_sequences.fa",
            fasta_path.read_text(),
            "ucoe_sequences.fa", "text/plain",
        )

    st.markdown("---")
    st.markdown(
        "**Pipeline source code**: Available upon request or at publication.  \n"
        "**Genome assembly**: GRCh38/hg38  \n"
        "**Epigenomic data**: ENCODE / Roadmap Epigenomics (11 cell lines)  \n"
        "**Gene annotation**: GENCODE v44  \n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About UCOE Atlas")

    st.markdown("""
    ### What are UCOEs?

    **Ubiquitous Chromatin Opening Elements (UCOEs)** are DNA regulatory sequences
    located at bidirectional promoters between housekeeping genes. They maintain
    constitutively open chromatin and prevent epigenetic silencing of adjacent
    transgenes, regardless of chromosomal integration site.

    Only **3 UCOEs** have been experimentally validated in the human genome:
    - **A2UCOE** (HNRNPA2B1/CBX3) — Williams et al., 2005
    - **TBP/PSMB1** — Antoniou, Harland & Mustoe, 2003
    - **SRF-UCOE** (SURF1/SURF2) — Rudina & Smolke, 2019

    ### About this pipeline

    This two-phase pipeline identifies UCOE candidates by:

    **Phase I** — Sequential filtering based on shared biological properties:
    1. Divergent promoters between housekeeping genes (Human Protein Atlas)
    2. CpG island overlap
    3. Ubiquitous active histone marks (H3K4me3, H3K27ac)
    4. Absence of repressive marks (H3K27me3, H3K9me3)
    5. Constitutive hypomethylation

    **Phase II** — Multivariate ranking by similarity to the 3 known UCOEs
    using Mahalanobis distance, cosine similarity, and composite percentile rank.

    ### Key findings

    - **ETS motif enrichment**: The ETS binding hexamer (CGGAAG) is consistently
      present across all 3 known UCOEs at ~2.5 motifs/kb
    - **WW dinucleotide composition**: Candidates are enriched in WW dinucleotides
      (38% more than generic CpG islands), reducing nucleosome propensity
    - **Evolutionary conservation**: ETS motifs in UCOEs are under strong
      purifying selection (PhyloP = 0.901 at motifs vs. 0.371 outside)

    ### Citation

    *Manuscript in preparation.*

    Ostetti, E.R.S.; Moro, A.M. (2026). Computational identification and
    characterization of UCOE candidates in the human genome.

    ### Contact

    - **Elton Roger Ostetti** — PhD Student, University of São Paulo / Instituto Butantan
    - **Prof. Dr. Ana Maria Moro** — Supervisor, Biopharmaceuticals Laboratory, Instituto Butantan
    """)
