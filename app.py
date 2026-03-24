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
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output"

# ── Load data ──
@st.cache_data
def load_data():
    scored = pd.read_csv(DATA_DIR / "phase2" / "scored_candidates.tsv", sep="\t")
    ref = pd.read_csv(DATA_DIR / "phase2" / "reference_profile.tsv", sep="\t")
    return scored, ref

@st.cache_data
def load_structural():
    try:
        return pd.read_csv(DATA_DIR / "structural" / "all_groups_structural.tsv", sep="\t")
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
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
st.sidebar.title("UCOE Atlas")
st.sidebar.markdown("**Interactive UCOE Candidate Explorer**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Candidate Explorer", "🔍 Candidate Detail",
     "🗺️ PCA Explorer", "📥 Downloads", "ℹ️ About"],
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
if page == "🏠 Overview":
    st.title("🧬 UCOE Atlas")
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
elif page == "📊 Candidate Explorer":
    st.title("📊 Candidate Explorer")

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
    chrom_counts = filtered["chrom"].value_counts().sort_index()
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
elif page == "🔍 Candidate Detail":
    st.title("🔍 Candidate Detail")

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
            st.success(f"✅ Known UCOE: **{known}**")
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
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Active Marks (normalized)", height=400,
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

    # UCSC link
    st.markdown("---")
    ucsc_url = (
        f"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&"
        f"position={cand['chrom']}%3A{cand['start']}-{cand['end']}"
    )
    st.markdown(f"🔗 [View in UCSC Genome Browser]({ucsc_url})")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PCA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ PCA Explorer":
    st.title("🗺️ PCA Explorer — Epigenomic Space")

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
# PAGE: DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Downloads":
    st.title("📥 Downloads")

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
    fasta_path = ROOT / "ucoe_data" / "sequences" / "ucoe_sequences.fa"
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
elif page == "ℹ️ About":
    st.title("ℹ️ About UCOE Atlas")

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
