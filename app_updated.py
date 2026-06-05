import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Gut Microbiome Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Better Styling
# -------------------------------
st.markdown("""
<style>
    /* Metric container */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        margin: 8px 6px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
        color: #2E86C1;
    }
    /* Page background - Force white background always */
    .main {
        background-color: white !important;
    }
    
    /* Overall page background */
    [data-testid="stAppViewContainer"] {
        background-color: white !important;
    }
    
    /* Root background */
    .stApp {
        background-color: white !important;
    }
    /* Sidebar styling */
    .css-1lcbmhc, [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* Title and header text visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    /* Body text */
    body, p, div, span {
        color: #333333 !important;
    }
    
    /* All text elements */
    * {
        color: #333333 !important;
    }
    
    /* Selectbox styling - target the select element itself */
    [data-baseweb="select"] {
        background-color: white !important;
    }
    [data-baseweb="select"] span, 
    [data-baseweb="select"] div {
        color: #1a1a1a !important;
    }
    
    /* Radio button text */
    .stRadio label, .stRadio span {
        color: #1a1a1a !important;
    }
    
    /* Checkbox text */
    .stCheckbox label, .stCheckbox span {
        color: #1a1a1a !important;
    }
    
    /* Slider label and text */
    .stSlider label, .stSlider span {
        color: #1a1a1a !important;
    }
    
    /* Selectbox label */
    .stSelectbox label {
        color: #1a1a1a !important;
    }
    
    /* Dropdown menu items text */
    [role="option"], 
    [role="listbox"] {
        color: #1a1a1a !important;
    }
    
    /* Navigation radio buttons */
    [data-testid="stRadio"] label {
        color: #1a1a1a !important;
    }
    
    /* All labels across the app */
    label {
        color: #1a1a1a !important;
    }
    
    /* DROPDOWN SELECTBOX - Most important styles */
    [data-baseweb="select"] > div {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    [data-baseweb="select"] input {
        color: #1a1a1a !important;
    }
    
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="select"] p {
        color: #1a1a1a !important;
    }
    
    /* Popover menu items for selectbox - CRITICAL */
    [role="option"] {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    [role="option"] > div,
    [role="option"] span,
    [role="option"] p {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    /* Listbox styling with white background */
    [role="listbox"] {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    [role="listbox"] span,
    [role="listbox"] div,
    [role="listbox"] p {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    /* Popover for dropdown - make it white */
    [data-testid="stPopoverContent"],
    [class*="baseweb-popover"] {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    [data-testid="stPopoverContent"] span,
    [data-testid="stPopoverContent"] div,
    [data-testid="stPopoverContent"] p,
    [class*="baseweb-popover"] span,
    [class*="baseweb-popover"] div,
    [class*="baseweb-popover"] p {
        background-color: white !important;
        color: #1a1a1a !important;
    }
    
    /* Smaller text for footers */
    .small {
        font-size:12px;
        color:#555;
    }
    /* Improve table header visibility */
    .stDataFrame thead th {
        background-color: #f1f5f9;
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🧬 Gut Microbiome Analysis Dashboard")
st.markdown("### Dynamic Nutrient Bioavailability Prediction Model")

# -------------------------------
# Load Data
# -------------------------------
import os

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "india_species_abundance_clean.tsv")
    try:
        otu = pd.read_csv(path, sep="\t", index_col=0)
        otu = otu.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return otu
    except Exception:
        return None

otu = load_data()
baseline_abundance = otu.mean(axis=1)

# -------------------------------
# Genus Mapping & Traits
# -------------------------------
def get_genus(species):
    # Extract genus from species name and convert to lowercase for matching
    if isinstance(species, str):
        genus = species.split("_")[0]
        return genus.lower()  # Convert to lowercase for consistent matching
    return "unknown"

species_list = baseline_abundance.index.tolist()
genus_map = {sp: get_genus(sp) for sp in species_list}

genus_traits = {
    "lactobacillus": {"SCFA":0.9,"pH_reduction":0.9,"Barrier_support":0.8,"Vitamin_Biosynthesis":0.6,"Siderophore":0.0},
    "bifidobacterium": {"SCFA":0.8,"pH_reduction":0.7,"Barrier_support":0.8,"Vitamin_Biosynthesis":0.7,"Siderophore":0.0},
    "faecalibacterium": {"SCFA":0.95,"pH_reduction":0.9,"Barrier_support":0.9,"Vitamin_Biosynthesis":0.4,"Siderophore":0.0},
    "roseburia": {"SCFA":0.9,"pH_reduction":0.8,"Barrier_support":0.7,"Vitamin_Biosynthesis":0.2,"Siderophore":0.0},
    "bacteroides": {"SCFA":0.6,"pH_reduction":0.3,"Barrier_support":0.5,"Vitamin_Biosynthesis":0.5,"Siderophore":0.1},
    "prevotella": {"SCFA":0.7,"pH_reduction":0.4,"Barrier_support":0.6,"Vitamin_Biosynthesis":0.3,"Siderophore":0.0},
    "streptococcus": {"SCFA":0.3,"pH_reduction":0.5,"Barrier_support":0.3,"Vitamin_Biosynthesis":0.3,"Siderophore":0.0},
    "clostridium": {"SCFA":0.8,"pH_reduction":0.6,"Barrier_support":0.4,"Vitamin_Biosynthesis":0.2,"Siderophore":0.0},
    "ruminococcus": {"SCFA":0.85,"pH_reduction":0.7,"Barrier_support":0.6,"Vitamin_Biosynthesis":0.3,"Siderophore":0.0},
    "escherichia": {"SCFA":0.1,"pH_reduction":0.0,"Barrier_support":0.1,"Vitamin_Biosynthesis":0.1,"Siderophore":0.9},
    "enterobacter": {"SCFA":0.1,"pH_reduction":0.0,"Barrier_support":0.1,"Vitamin_Biosynthesis":0.1,"Siderophore":0.8},
    "eubacterium": {"SCFA":0.5,"pH_reduction":0.4,"Barrier_support":0.3,"Vitamin_Biosynthesis":0.2,"Siderophore":0.0},
    "haemophilus": {"SCFA":0.2,"pH_reduction":0.2,"Barrier_support":0.2,"Vitamin_Biosynthesis":0.2,"Siderophore":0.3},
    "megasphaera": {"SCFA":0.7,"pH_reduction":0.5,"Barrier_support":0.4,"Vitamin_Biosynthesis":0.1,"Siderophore":0.0},
    "parasutterella": {"SCFA":0.3,"pH_reduction":0.2,"Barrier_support":0.2,"Vitamin_Biosynthesis":0.2,"Siderophore":0.2},
    "gemmiger": {"SCFA":0.6,"pH_reduction":0.4,"Barrier_support":0.5,"Vitamin_Biosynthesis":0.3,"Siderophore":0.0}
}

# Build species-level trait matrix (species x traits)
default_trait = {"SCFA":0.2,"pH_reduction":0.1,"Barrier_support":0.1,"Vitamin_Biosynthesis":0.1,"Siderophore":0.2}
all_traits = []
for sp in species_list:
    genus = genus_map.get(sp, None)
    row = genus_traits.get(genus, default_trait).copy()
    row["Species"] = sp
    all_traits.append(row)
traits = pd.DataFrame(all_traits).set_index("Species")

# -------------------------------
# Nutrient model (configurable)
# -------------------------------
nutrients = {
    "Iron": {"SCFA":0.4,"pH_reduction":0.3,"Barrier_support":0.2,"Siderophore":-0.6},
    "Vitamin_B12": {"Vitamin_Biosynthesis":0.7,"Barrier_support":0.2},
    "Folate": {"Vitamin_Biosynthesis":0.6},
    "Calcium": {"SCFA":0.5,"pH_reduction":0.4},
    "Magnesium": {"SCFA":0.4},
    "Zinc": {"Barrier_support":0.4,"Siderophore":-0.5}
}

def absorption_score(abundance, nutrient, normalize=True):
    """
    Compute absorption score for a nutrient given species abundance (pd.Series or dict).
    If normalize=True, divide by total abundance to get per-unit effect.
    """
    if isinstance(abundance, pd.Series):
        abundance = abundance.to_dict()
    score = 0.0
    for sp, ab in abundance.items():
        if sp in traits.index and ab is not None and ab > 0:
            for trait, w in nutrients[nutrient].items():
                if trait in traits.columns:
                    score += ab * traits.loc[sp, trait] * w
    if normalize:
        total_ab = sum([v for v in abundance.values() if v is not None])
        if total_ab > 0:
            return score / total_ab
        return 0.0
    return score

# -------------------------------
# Sidebar Navigation
# -------------------------------
# REPLACE with this:
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "OTU Analysis", "Trait Coverage", "Nutrient Absorption", "Simulations",
     "🔗 Microbial Ecology & T2D",
     "About"],
    index=0
)

# -------------------------------
# Overview Page
# -------------------------------
if page == "Overview":
    st.header("🧬 Project Overview")

    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("🦠 Species Detected", len(baseline_abundance))
        c2.metric("📊 Samples Analyzed", otu.shape[1])
        c3.metric("🔬 Nutrient Models", len(nutrients))

    st.markdown("---")
    st.subheader("📋 About This Dashboard")
    st.write("""
    This microbiome analysis platform integrates:
    - **OTU Composition Analysis**
    - **Nutrient Bioavailability Modeling**
    - **Interactive Simulations**
    - **Functional Trait Mapping**
    """)

    st.markdown("---")
    st.subheader("🌟 Top 10 Most Abundant Species")
    top_species = baseline_abundance.sort_values(ascending=False).head(10)
    fig = px.bar(
        x=top_species.values,
        y=top_species.index,
        orientation='h',
        labels={"x": "Mean Relative Abundance", "y": "Species"},
        color=top_species.values,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=520, margin=dict(l=40,r=40,t=60,b=40))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# OTU Analysis Page
# -------------------------------
elif page == "OTU Analysis":
    st.header("OTU Abundance Analysis")

    st.subheader("Abundance Distribution")
    col1, col2 = st.columns([1,1])
    with col1:
        fig = px.histogram(
            baseline_abundance,
            nbins=60,
            title="Distribution of Mean Species Abundance",
            labels={"value": "Mean Relative Abundance"},
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(height=420, margin=dict(l=40,r=40,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cumulative Abundance")
        sorted_abund = baseline_abundance.sort_values(ascending=False)
        cumsum = np.cumsum(sorted_abund.values)
        cumsum_pct = (cumsum / cumsum[-1]) * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=cumsum_pct,
            mode='lines+markers',
            fill='tozeroy',
            name='Cumulative %',
            line=dict(color='#00CC96')
        ))
        fig2.update_layout(title="Cumulative Species Abundance", xaxis_title="Species (ranked)", yaxis_title="Cumulative Abundance (%)", height=420, margin=dict(l=40,r=40,t=60,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Species Data Table")
    display_df = pd.DataFrame({
        "Species": baseline_abundance.index,
        "Mean Abundance": baseline_abundance.values,
        "Genus": [genus_map.get(sp, "Unknown") for sp in baseline_abundance.index]
    }).sort_values("Mean Abundance", ascending=False).reset_index(drop=True)
    st.dataframe(display_df.style.background_gradient(subset=["Mean Abundance"], cmap="Blues"), use_container_width=True, height=420)

# -------------------------------
# Trait Coverage Page
# -------------------------------
elif page == "Trait Coverage":
    st.header("Trait Coverage & Diagnostics")
    st.write("Visualize how traits are distributed across species and check coverage.")

    # Heatmap of traits (transpose for readability)
    trait_matrix = traits.copy()
    if trait_matrix.shape[0] > 0:
        fig = px.imshow(
            trait_matrix.T,
            labels=dict(x="Species", y="Trait", color="Value"),
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=520, margin=dict(l=40,r=40,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Species by Trait (example: SCFA)")
    top_scfa = traits["SCFA"].sort_values(ascending=False).head(15)
    fig2 = px.bar(x=top_scfa.values, y=top_scfa.index, orientation='h', labels={"x":"SCFA trait value","y":"Species"}, color=top_scfa.values, color_continuous_scale="Blues")
    fig2.update_layout(height=420, margin=dict(l=40,r=40,t=60,b=40))
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Nutrient Absorption Page
# -------------------------------
elif page == "Nutrient Absorption":
    st.header("Nutrient Bioavailability Analysis")

    # Baseline nutrient scores (normalized)
    nutrient_scores = {n: absorption_score(baseline_abundance, n, normalize=True) for n in nutrients.keys()}

    left, right = st.columns([1,2])
    with left:
        st.subheader("Baseline Absorption Scores")
        for nutrient, score in sorted(nutrient_scores.items(), key=lambda x: x[1], reverse=True):
            st.metric(nutrient, f"{score:.4f}")

    with right:
        st.subheader("Nutrient Comparison")
        fig = px.bar(
            x=list(nutrient_scores.keys()),
            y=list(nutrient_scores.values()),
            title="Baseline Nutrient Absorption Scores (normalized)",
            labels={"x":"Nutrient","y":"Absorption Score"},
            color=list(nutrient_scores.values()),
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=520, margin=dict(l=40,r=40,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Nutrient-Trait Relationships")
    selected_nutrient = st.selectbox("Select Nutrient:", list(nutrients.keys()))

    # Compute per-species contribution (trait-weight only, independent of abundance)
    contrib = {}
    for sp in traits.index:
        score = 0.0
        for trait, weight in nutrients[selected_nutrient].items():
            if trait in traits.columns:
                score += traits.loc[sp, trait] * weight
        # Store all contributions, even if small
        contrib[sp] = max(score, 0)  # Use 0 if negative
    
    contrib_df = pd.DataFrame(list(contrib.items()), columns=["Species","Contribution"])
    contrib_df = contrib_df.sort_values("Contribution", ascending=False)
    # Show top species with contributions, filter only if exactly 0
    contrib_df = contrib_df[contrib_df["Contribution"] > 0].head(15)
    
    if len(contrib_df) > 0:
        fig3 = px.bar(contrib_df, x="Contribution", y="Species", orientation='h', 
                      color="Contribution", color_continuous_scale="RdBu")
        fig3.update_layout(height=520, margin=dict(l=40,r=40,t=60,b=40))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No species with positive contributions for this nutrient.")

# -------------------------------
# Simulations Page
# -------------------------------
elif page == "Simulations":
    st.header("Microbiome Perturbation Simulator")
    st.write("Simulate the impact of adding or removing specific bacterial species on nutrient absorption (normalized scores).")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_bacteria = st.selectbox("Select Bacteria:", sorted(list(set(baseline_abundance.index).intersection(traits.index))))
    with col2:
        selected_nutrient = st.selectbox("Select Nutrient:", list(nutrients.keys()), key="sim_nutrient")
    with col3:
        delta = st.slider("Δ Abundance (additive):", min_value=-0.2, max_value=0.5, value=0.05, step=0.01)

    def simulate_addition(abundance, microbe, delta):
        new = abundance.copy()
        if microbe in new:
            new[microbe] = max(0.0, new[microbe] + delta)
        else:
            new[microbe] = max(0.0, delta)
        return new

    baseline_score = absorption_score(baseline_abundance, selected_nutrient, normalize=True)
    new_abundance = simulate_addition(baseline_abundance.to_dict(), selected_bacteria, delta)
    new_score = absorption_score(new_abundance, selected_nutrient, normalize=True)
    change = new_score - baseline_score
    pct_change = (change / baseline_score * 100) if baseline_score != 0 else np.nan

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Score", f"{baseline_score:.4f}")
    m2.metric("After Perturbation", f"{new_score:.4f}")
    
    # Color-coded impact metric: green for positive, red for negative
    impact_color = "🟢" if change >= 0 else "🔴"
    m3.metric("Change (Δ)", f"{change:+.4f}", delta=change, delta_color="inverse")
    
    # Display colored impact summary
    if change > 0:
        st.success(f"✅ **Positive Impact**: +{change:.4f} ({pct_change:+.1f}%)")
    elif change < 0:
        st.error(f"❌ **Negative Impact**: {change:.4f} ({pct_change:+.1f}%)")
    else:
        st.info(f"ℹ️ **No Change**: {change:.4f} (0.0%)")

    st.markdown("---")
    chart_col, summary_col = st.columns([1.6, 1])
    with chart_col:
        comparison_data = pd.DataFrame({
            "Condition": ["Baseline", "Perturbed"],
            "Score": [baseline_score, new_score]
        })
        fig = px.bar(comparison_data, x="Condition", y="Score", color="Score", color_continuous_scale="RdYlGn", text="Score")
        fig.update_traces(texttemplate="%{text:.4f}", textposition='outside')
        fig.update_layout(height=480, margin=dict(l=40,r=40,t=60,b=40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with summary_col:
        st.info(f"""
**Simulation Summary**

**Bacteria:** {selected_bacteria}

**Δ Abundance:** {delta:+.2f}

**Nutrient:** {selected_nutrient}

**Impact:** {change:+.4f} ({pct_change:+.1f}%)
""")

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.header("About This Project")
    st.subheader("Overview")
    st.write("""
    This microbiome analysis dashboard models the relationship between gut bacterial composition 
    and nutrient bioavailability using functional traits and interaction networks.
    """)
    st.subheader("Methodology")
    st.write("""
    **Data Processing:**
    - OTU tables normalized to relative abundance
    - Spearman correlation networks for co-occurrence patterns

    **Functional Traits:**
    - Genus-level priors based on published literature
    - SCFA production, pH reduction, barrier support, vitamin biosynthesis, siderophore production

    **Nutrient Model:**
    - Trait-to-nutrient mapping through trait weight relationships
    - Absorption scores reflect potential nutrient bioavailability

    **Simulations:**
    - Real-time response to microbiome perturbations
    """)
    st.subheader("Data Requirements")
    st.write("- `india_species_abundance_clean.tsv` - OTU abundance table (species × samples)")
    st.subheader("Built With")
    st.write("🐍 Python | 📊 Streamlit | 📈 Plotly | 🔬 SciPy")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown('<div class="small">🧬 Gut Microbiome Analysis Dashboard | Built with Streamlit</div>', unsafe_allow_html=True)


# ============================================================
# DIABETES & GUT ECOLOGY ADDON — Community Imbalance Focus
#
# Scientific framing:
#   Diabetes is not caused by one microbe.
#   It emerges when the microbial COMMUNITY loses balance —
#   keystone species decline, opportunists bloom, and the
#   collective metabolic output of the gut shifts in a way
#   that promotes inflammation and insulin resistance.
#
# This module shows:
#   1. Who lives in the gut and how they interact
#   2. Which interactions are disrupted in T2D-associated imbalance
#   3. What that imbalance does metabolically
#   4. The causal chain from microbial ecology to glucose dysregulation
#
# Integration:
#   1. Add "🔗 Microbial Ecology & T2D" to sidebar radio list
#   2. Paste elif block at bottom of page routing section
#   3. No changes to existing code needed
# ============================================================

import networkx as nx

# ---------------------------------------------------------------
# KNOWN MICROBIAL INTERACTION TYPES (literature-derived)
# Used to annotate correlation network edges
# Positive correlation = cooperation / shared niche
# Negative correlation = competition / antagonism
# ---------------------------------------------------------------
known_interactions = {
    ("faecalibacterium", "roseburia"):   "Syntrophic (butyrate cross-feeding)",
    ("faecalibacterium", "bifidobacterium"): "Cooperative (lactate→butyrate)",
    ("bifidobacterium",  "lactobacillus"):   "Cooperative (acidification)",
    ("akkermansia",      "faecalibacterium"):"Mucosal niche sharing",
    ("akkermansia",      "bacteroides"):     "Competition (mucus layer)",
    ("escherichia",      "lactobacillus"):   "Antagonistic (pH competition)",
    ("escherichia",      "bifidobacterium"): "Antagonistic",
    ("clostridium",      "faecalibacterium"):"Competition",
    ("bacteroides",      "prevotella"):      "Competition (Bacteroidetes niche)",
    ("streptococcus",    "lactobacillus"):   "Competition (lactate niche)",
    ("desulfovibrio",    "faecalibacterium"):"Antagonistic (H₂S vs butyrate)",
}

# ---------------------------------------------------------------
# T2D COMMUNITY SIGNATURES — framed as IMBALANCE PATTERNS
# Not "bad species" but "when this balance breaks, here's what fails"
# ---------------------------------------------------------------
t2d_imbalance_patterns = {
    "Butyrate Producer Collapse": {
        "depleted_genera":  ["faecalibacterium","roseburia","coprococcus","blautia"],
        "enriched_genera":  ["clostridium","escherichia","streptococcus"],
        "metabolic_consequence": "SCFA output drops → colonocyte fuel deficit → gut barrier weakens → LPS enters bloodstream",
        "glycaemic_link":  "Circulating LPS activates TLR4 on adipocytes → NF-κB → TNF-α → insulin receptor phosphorylation blocked → insulin resistance",
        "severity": "HIGH"
    },
    "Mucosal Layer Erosion": {
        "depleted_genera":  ["akkermansia","bifidobacterium"],
        "enriched_genera":  ["desulfovibrio","bacteroides"],
        "metabolic_consequence": "Mucin layer thins → bacteria translocate → systemic endotoxemia",
        "glycaemic_link":  "Endotoxemia triggers chronic low-grade inflammation → impairs GLP-1 signalling → post-prandial hyperglycaemia",
        "severity": "HIGH"
    },
    "Diversity Collapse": {
        "depleted_genera":  ["prevotella","ruminococcus","eubacterium","gemmiger"],
        "enriched_genera":  ["escherichia","enterobacter"],
        "metabolic_consequence": "Functional redundancy lost → community fragile to perturbation → metabolic flexibility reduced",
        "glycaemic_link":  "Less diverse bile acid biotransformation → impaired FXR/TGR5 signalling → reduced incretin secretion → glucose intolerance",
        "severity": "MODERATE"
    },
    "Lactate Accumulation": {
        "depleted_genera":  ["faecalibacterium","blautia"],
        "enriched_genera":  ["streptococcus","lactobacillus"],   # excess lactobacillus without downstream consumers
        "metabolic_consequence": "Lactate overproduction without butyrate-converting consumers → gut acidosis",
        "glycaemic_link":  "Acidic gut environment → altered nutrient absorption kinetics → post-meal glucose spikes",
        "severity": "MODERATE"
    },
}

# ---------------------------------------------------------------
# CAUSAL CHAIN STEPS (for visual flow diagram)
# ---------------------------------------------------------------
causal_chain = [
    {
        "step": 1, "level": "Ecology",
        "title": "Community Imbalance",
        "detail": "Keystone butyrate producers (Faecalibacterium, Roseburia) decline. Gram-negative opportunists bloom.",
        "color": "#EF5350"
    },
    {
        "step": 2, "level": "Microbial Metabolism",
        "title": "Metabolic Output Shifts",
        "detail": "SCFA (butyrate, propionate) production falls. LPS, H₂S, and secondary bile acids increase.",
        "color": "#FF7043"
    },
    {
        "step": 3, "level": "Gut Physiology",
        "title": "Barrier Integrity Fails",
        "detail": "Butyrate deficit → tight junction proteins degrade → intestinal permeability increases ('leaky gut').",
        "color": "#FFA726"
    },
    {
        "step": 4, "level": "Systemic",
        "title": "Endotoxemia & Inflammation",
        "detail": "LPS and bacterial fragments enter portal circulation → TLR4 activation → TNF-α, IL-6, IL-1β surge.",
        "color": "#FFCA28"
    },
    {
        "step": 5, "level": "Metabolic",
        "title": "Insulin Signalling Disrupted",
        "detail": "Cytokines phosphorylate IRS-1 at inhibitory sites → downstream PI3K/Akt pathway impaired → insulin resistance.",
        "color": "#66BB6A"
    },
    {
        "step": 6, "level": "Clinical",
        "title": "Glucose Dysregulation → T2D Risk",
        "detail": "Hepatic glucose output uncontrolled. Peripheral uptake impaired. HbA1c rises progressively.",
        "color": "#42A5F5"
    },
]

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------

def compute_spearman_network(otu_df, top_n=30, threshold=0.4):
    """
    Compute Spearman correlation matrix across species (rows) × samples (cols).
    Returns filtered edge list for network display.
    top_n: use top N most abundant species (keeps network readable)
    threshold: minimum |r| to draw an edge
    """
    top_species = otu_df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    sub = otu_df.loc[top_species].T  # samples × species

    corr_matrix = pd.DataFrame(index=top_species, columns=top_species, dtype=float)
    pval_matrix = pd.DataFrame(index=top_species, columns=top_species, dtype=float)

    for i, sp1 in enumerate(top_species):
        for j, sp2 in enumerate(top_species):
            if i == j:
                corr_matrix.loc[sp1, sp2] = 1.0
                pval_matrix.loc[sp1, sp2] = 0.0
            elif i < j:
                r, p = spearmanr(sub[sp1], sub[sp2])
                corr_matrix.loc[sp1, sp2] = r
                corr_matrix.loc[sp2, sp1] = r
                pval_matrix.loc[sp1, sp2] = p
                pval_matrix.loc[sp2, sp1] = p

    edges = []
    for i, sp1 in enumerate(top_species):
        for j, sp2 in enumerate(top_species):
            if i < j:
                r = corr_matrix.loc[sp1, sp2]
                p = pval_matrix.loc[sp1, sp2]
                if abs(r) >= threshold and p < 0.05:
                    edges.append({
                        "source": sp1, "target": sp2,
                        "r": round(float(r), 3), "p": round(float(p), 4),
                        "type": "positive" if r > 0 else "negative"
                    })
    return edges, corr_matrix


def compute_community_metrics(abundance_series, genus_map_dict, otu_df):
    """Community-level balance metrics."""
    # Shannon diversity
    ab = abundance_series[abundance_series > 0]
    p = ab / ab.sum()
    shannon = float(-np.sum(p * np.log(p)))

    # Firmicutes:Bacteroidetes ratio (F:B ratio)
    firmicutes_genera  = {"lactobacillus","clostridium","ruminococcus","roseburia",
                           "faecalibacterium","coprococcus","blautia","streptococcus","eubacterium"}
    bacteroidetes_genera = {"bacteroides","prevotella"}

    firm_ab  = sum(v for sp, v in abundance_series.items()
                   if genus_map_dict.get(sp,"") in firmicutes_genera)
    bact_ab  = sum(v for sp, v in abundance_series.items()
                   if genus_map_dict.get(sp,"") in bacteroidetes_genera)
    fb_ratio = firm_ab / bact_ab if bact_ab > 0 else float("inf")

    # Butyrate producer ratio
    butyrate_genera = {"faecalibacterium","roseburia","coprococcus","blautia",
                        "eubacterium","ruminococcus"}
    butyrate_ab = sum(v for sp, v in abundance_series.items()
                      if genus_map_dict.get(sp,"") in butyrate_genera)
    butyrate_ratio = butyrate_ab / abundance_series.sum() if abundance_series.sum() > 0 else 0

    # Pathobiont load
    pathobiont_genera = {"escherichia","clostridium","enterobacter","desulfovibrio","streptococcus"}
    pathobiont_ab = sum(v for sp, v in abundance_series.items()
                        if genus_map_dict.get(sp,"") in pathobiont_genera)
    pathobiont_ratio = pathobiont_ab / abundance_series.sum() if abundance_series.sum() > 0 else 0

    return {
        "Shannon Diversity": round(shannon, 3),
        "F:B Ratio": round(fb_ratio, 3),
        "Butyrate Producer Ratio": round(butyrate_ratio, 4),
        "Pathobiont Load": round(pathobiont_ratio, 4),
        "Species Richness": int((abundance_series > 0).sum()),
    }


# ---------------------------------------------------------------
# PAGE ROUTING — add this block to the main routing section
# ---------------------------------------------------------------

if 'page' in globals() and page == "🔗 Microbial Ecology & T2D":

    st.header("🔗 Microbial Community Ecology & Diabetes")
    st.markdown("""
    > **The core question this module answers:**  
    > *Who lives in this gut, how do they interact with each other, and when that balance breaks — 
    > what does that mean for glucose regulation?*  
    >  
    > Diabetes is not caused by a single microbe. It emerges when the **community structure collapses** — 
    > keystone species that anchor the network decline, opportunists fill the vacuum, and the collective 
    > metabolic output of the gut shifts away from barrier protection and short-chain fatty acid production 
    > toward endotoxin release and inflammation.
    """)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Interaction Network",
        "⚖️ Community Balance",
        "🔥 Imbalance Patterns",
        "🔬 Causal Chain"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — INTERACTION NETWORK
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Microbial Co-occurrence & Interaction Network")
        st.write("""
        Built from **Spearman correlations** across your actual sample data.  
        - 🟢 **Green edges** = positive correlation (co-occur together; cooperative or shared niche)  
        - 🔴 **Red edges** = negative correlation (mutually exclusive; competitive or antagonistic)  
        - **Node size** = mean relative abundance  
        - **Node colour** = genus functional group  
        Hover over nodes and edges for details.
        """)

        net_col1, net_col2 = st.columns([1, 3])
        with net_col1:
            top_n    = st.slider("Top N species to include:", 10, 50, 25, 5)
            threshold = st.slider("Correlation threshold |r| ≥:", 0.2, 0.8, 0.4, 0.05)
            st.caption("Higher threshold = fewer, stronger edges only")

        with net_col2:
            with st.spinner("Computing Spearman correlations across samples..."):
                edges, corr_matrix = compute_spearman_network(otu, top_n=top_n, threshold=threshold)

            if not edges:
                st.warning("No significant correlations found at this threshold. Try lowering it.")
            else:
                # Build networkx graph for layout
                G = nx.Graph()
                top_species_net = otu.mean(axis=1).sort_values(ascending=False).head(top_n).index.tolist()
                ab_vals = baseline_abundance[top_species_net]

                for sp in top_species_net:
                    G.add_node(sp)
                for e in edges:
                    G.add_edge(e["source"], e["target"], weight=abs(e["r"]), r=e["r"])

                # Spring layout
                pos = nx.spring_layout(G, seed=42, k=2.5)

                # Genus colour mapping
                genus_color_map = {
                    "faecalibacterium": "#4CAF50","roseburia":    "#66BB6A",
                    "bifidobacterium":  "#2196F3","lactobacillus":"#42A5F5",
                    "akkermansia":      "#00BCD4","blautia":      "#26C6DA",
                    "bacteroides":      "#FF9800","prevotella":   "#FFA726",
                    "clostridium":      "#F44336","escherichia":  "#EF5350",
                    "streptococcus":    "#FF7043","desulfovibrio":"#B71C1C",
                    "ruminococcus":     "#8BC34A","eubacterium":  "#CDDC39",
                    "coprococcus":      "#009688",
                }
                default_color = "#9E9E9E"

                # Edge traces
                edge_traces = []
                for e in edges:
                    if e["source"] in pos and e["target"] in pos:
                        x0, y0 = pos[e["source"]]
                        x1, y1 = pos[e["target"]]
                        color = "rgba(76,175,80,0.6)" if e["type"] == "positive" else "rgba(244,67,54,0.6)"
                        width = max(1, abs(e["r"]) * 5)
                        interaction_label = known_interactions.get(
                            (genus_map.get(e["source"],""), genus_map.get(e["target"],"")),
                            known_interactions.get(
                                (genus_map.get(e["target"],""), genus_map.get(e["source"],"")),
                                "Unknown interaction"
                            )
                        )
                        edge_traces.append(go.Scatter(
                            x=[x0, x1, None], y=[y0, y1, None],
                            mode="lines",
                            line=dict(width=width, color=color),
                            hoverinfo="text",
                            text=f"r = {e['r']:.3f} | p = {e['p']:.4f}<br>{interaction_label}",
                            showlegend=False
                        ))

                # Node trace
                node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                for sp in top_species_net:
                    if sp in pos:
                        x, y = pos[sp]
                        node_x.append(x)
                        node_y.append(y)
                        g = genus_map.get(sp, "unknown")
                        ab = float(ab_vals.get(sp, 0))
                        degree = G.degree(sp)
                        node_text.append(
                            f"<b>{sp}</b><br>Genus: {g}<br>"
                            f"Abundance: {ab:.4f}<br>Connections: {degree}"
                        )
                        node_color.append(genus_color_map.get(g, default_color))
                        node_size.append(max(8, min(40, ab * 2000 + 12)))

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text",
                    text=[sp.split("_")[0][:10] for sp in top_species_net if sp in pos],
                    textposition="top center",
                    textfont=dict(size=8),
                    hoverinfo="text",
                    hovertext=node_text,
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=1.5, color="white")
                    ),
                    showlegend=False
                )

                fig_net = go.Figure(data=edge_traces + [node_trace])
                fig_net.update_layout(
                    title=f"Co-occurrence Network: Top {top_n} species | {len(edges)} significant edges (|r| ≥ {threshold})",
                    showlegend=False,
                    hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=580,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor="white"
                )
                st.plotly_chart(fig_net, use_container_width=True)

                # Keystone species (highest degree nodes)
                st.markdown("---")
                st.subheader("🔑 Keystone Species (Hub Nodes)")
                st.write("""
                Keystones are the most *connected* species in the network. 
                Losing them doesn't just remove one species — it destabilizes the 
                entire neighbourhood of species that depend on their metabolic output.
                """)
                degrees = dict(G.degree())
                degree_df = pd.DataFrame([
                    {
                        "Species": sp,
                        "Genus": genus_map.get(sp, "?").capitalize(),
                        "Network Connections": degrees.get(sp, 0),
                        "Mean Abundance": round(float(baseline_abundance.get(sp, 0)), 6),
                        "Role": ("⚠️ Keystone — high connectivity" 
                                 if degrees.get(sp, 0) >= sorted(degrees.values())[-max(1, len(degrees)//5)]
                                 else "Peripheral")
                    }
                    for sp in top_species_net
                ]).sort_values("Network Connections", ascending=False).head(10)

                st.dataframe(degree_df, use_container_width=True, hide_index=True)

                # Correlation heatmap
                st.markdown("---")
                st.subheader("Correlation Heatmap")
                fig_heat = px.imshow(
                    corr_matrix.astype(float),
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    zmin=-1, zmax=1,
                    labels=dict(color="Spearman r"),
                    title="Species–Species Spearman Correlation Matrix"
                )
                fig_heat.update_layout(height=560, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig_heat, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — COMMUNITY BALANCE
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Community Balance Metrics")
        st.write("""
        These metrics capture the *shape* of the community, not individual species.
        A community can have the "right" species present but still be imbalanced 
        if proportions are skewed.
        """)

        metrics = compute_community_metrics(baseline_abundance, genus_map, otu)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Shannon Diversity", metrics["Shannon Diversity"],
                  help="Higher = more diverse = more resilient community")
        m2.metric("F:B Ratio", metrics["F:B Ratio"],
                  help="Firmicutes:Bacteroidetes. T2D often shows elevated F:B ratio")
        m3.metric("Butyrate Producer Ratio", f"{metrics['Butyrate Producer Ratio']:.4f}",
                  help="Fraction of community that produces butyrate — key metabolite for gut barrier")
        m4.metric("Pathobiont Load", f"{metrics['Pathobiont Load']:.4f}",
                  help="Fraction of community composed of T2D-associated opportunistic genera")
        m5.metric("Species Richness", metrics["Species Richness"],
                  help="Total number of detectable species")

        st.markdown("---")

        # Visual balance summary
        st.subheader("What does imbalanced vs. balanced look like?")

        balance_data = {
            "Metric": ["Shannon Diversity", "Butyrate Producers", "Pathobiont Load", "F:B Ratio (adjusted)"],
            "Your Cohort": [
                metrics["Shannon Diversity"] / 5.0,              # normalise to 0-1
                metrics["Butyrate Producer Ratio"] * 20,
                1 - metrics["Pathobiont Load"] * 20,             # invert (lower is better)
                min(1.0, 1 / max(metrics["F:B Ratio"], 0.1))     # invert & clamp
            ],
            "Healthy Reference": [0.8, 0.7, 0.85, 0.6],
            "T2D Pattern":       [0.4, 0.25, 0.35, 0.3]
        }
        balance_df = pd.DataFrame(balance_data)

        fig_radar = go.Figure()
        categories = balance_data["Metric"]
        for col, color, dash in [
            ("Your Cohort", "#2196F3", "solid"),
            ("Healthy Reference", "#4CAF50", "dot"),
            ("T2D Pattern", "#F44336", "dash")
        ]:
            vals = balance_df[col].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=col,
                line=dict(color=color, dash=dash, width=2.5),
                fill="toself" if col == "Your Cohort" else "none",
                fillcolor="rgba(33,150,243,0.1)" if col == "Your Cohort" else "rgba(0,0,0,0)"
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True,
            title="Community Balance Profile vs. Reference Patterns",
            height=500,
            margin=dict(l=40,r=40,t=60,b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("All axes normalised to 0–1 for comparability. Your cohort vs healthy and T2D community patterns.")

        # Per-sample diversity (if multiple samples)
        if otu.shape[1] > 1:
            st.markdown("---")
            st.subheader("Per-Sample Shannon Diversity")
            sample_diversity = []
            for col in otu.columns:
                ab = otu[col].dropna()
                ab = ab[ab > 0]
                p = ab / ab.sum()
                h = float(-np.sum(p * np.log(p)))
                sample_diversity.append({"Sample": col, "Shannon Diversity": round(h, 3)})
            sdiv_df = pd.DataFrame(sample_diversity).sort_values("Shannon Diversity", ascending=False)
            fig_div = px.bar(
                sdiv_df, x="Sample", y="Shannon Diversity",
                color="Shannon Diversity", color_continuous_scale="RdYlGn",
                title="Shannon Diversity per Sample"
            )
            fig_div.add_hline(y=sdiv_df["Shannon Diversity"].mean(), 
                              line_dash="dash", annotation_text="Mean")
            fig_div.update_layout(height=380, margin=dict(l=20,r=20,t=50,b=80),
                                  xaxis_tickangle=-45)
            st.plotly_chart(fig_div, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — IMBALANCE PATTERNS
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.subheader("T2D-Associated Imbalance Patterns")
        st.write("""
        These are **ecological patterns** — not "bad bacteria" — but combinations 
        of depletion and enrichment that collectively shift the gut's metabolic output 
        toward promoting insulin resistance. Each pattern represents a well-characterised 
        community-level disruption from published metagenomics cohorts.
        """)

        for pattern_name, pattern in t2d_imbalance_patterns.items():
            severity_color = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}
            with st.expander(
                f"{severity_color.get(pattern['severity'],'⚪')} **{pattern_name}** — {pattern['severity']} severity"
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Depleted in this pattern:**")
                    for g in pattern["depleted_genera"]:
                        g_ab = sum(v for sp, v in baseline_abundance.items()
                                   if genus_map.get(sp,"") == g)
                        status = "✅ Present" if g_ab > 0 else "❌ Absent/undetected"
                        st.markdown(f"- *{g.capitalize()}* — {status} (abundance: `{g_ab:.5f}`)")

                with c2:
                    st.markdown("**Enriched in this pattern:**")
                    for g in pattern["enriched_genera"]:
                        g_ab = sum(v for sp, v in baseline_abundance.items()
                                   if genus_map.get(sp,"") == g)
                        level = "⚠️ Detected" if g_ab > 0 else "✅ Low/absent"
                        st.markdown(f"- *{g.capitalize()}* — {level} (abundance: `{g_ab:.5f}`)")

                st.markdown("---")
                st.markdown(f"**What fails metabolically:** {pattern['metabolic_consequence']}")
                st.markdown(f"**Glycaemic link:** {pattern['glycaemic_link']}")

        # Pattern match summary
        st.markdown("---")
        st.subheader("Pattern Match Summary")
        st.write("How closely does your cohort's composition match each imbalance pattern?")

        match_scores = []
        for pname, pat in t2d_imbalance_patterns.items():
            depletion_hits = sum(
                1 for g in pat["depleted_genera"]
                if sum(v for sp,v in baseline_abundance.items() if genus_map.get(sp,"") == g) < 1e-4
            )
            enrichment_hits = sum(
                1 for g in pat["enriched_genera"]
                if sum(v for sp,v in baseline_abundance.items() if genus_map.get(sp,"") == g) > 1e-4
            )
            total = len(pat["depleted_genera"]) + len(pat["enriched_genera"])
            match_pct = (depletion_hits + enrichment_hits) / total * 100
            match_scores.append({
                "Pattern": pname,
                "Match %": round(match_pct, 1),
                "Severity": pat["severity"]
            })

        match_df = pd.DataFrame(match_scores).sort_values("Match %", ascending=False)
        fig_match = px.bar(
            match_df, x="Match %", y="Pattern", orientation="h",
            color="Match %", color_continuous_scale="RdYlGn_r",
            range_x=[0, 100],
            title="Cohort Similarity to T2D Imbalance Patterns (%)"
        )
        fig_match.add_vline(x=50, line_dash="dash", annotation_text="50% threshold")
        fig_match.update_layout(height=300, margin=dict(l=20,r=40,t=50,b=20))
        st.plotly_chart(fig_match, use_container_width=True)
        st.caption("A high match % means your cohort's absent/present genera align with that dysbiosis pattern.")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — CAUSAL CHAIN
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.subheader("From Microbial Imbalance to T2D: The Causal Chain")
        st.write("""
        This is the mechanistic story — how an ecological shift in your gut 
        community propagates through physiology into glucose dysregulation.
        Each step is supported by experimental and epidemiological evidence.
        """)

        for step in causal_chain:
            with st.container():
                col_num, col_content = st.columns([1, 9])
                with col_num:
                    st.markdown(
                        f"<div style='background:{step['color']};color:white;border-radius:50%;width:44px;"
                        f"height:44px;display:flex;align-items:center;justify-content:center;"
                        f"font-size:20px;font-weight:bold;margin-top:4px;'>{step['step']}</div>",
                        unsafe_allow_html=True
                    )
                with col_content:
                    st.markdown(f"**{step['level']} level — {step['title']}**")
                    st.write(step["detail"])
                if step["step"] < len(causal_chain):
                    st.markdown(
                        "<div style='text-align:center;font-size:22px;color:#9E9E9E;'>↓</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("---")
        st.subheader("Key Metabolites Linking Microbiome to Glucose Control")
        metabolite_data = {
            "Metabolite": ["Butyrate","Propionate","Acetate","LPS (endotoxin)","H₂S","Secondary Bile Acids","Indole-3-propionic acid"],
            "Produced by": ["Faecalibacterium, Roseburia, Coprococcus","Bacteroides, Prevotella, Ruminococcus",
                            "Bifidobacterium, Blautia","Escherichia, gram-negative bacteria",
                            "Desulfovibrio","Bacteroides, Clostridium (transformation)","Lactobacillus, Clostridium"],
            "Effect on glucose metabolism": [
                "✅ Colonocyte fuel; tight junction protein expression; GLP-1 stimulation",
                "✅ Hepatic gluconeogenesis substrate; satiety via PYY; reduces hepatic lipogenesis",
                "✅ Peripheral energy substrate; signalling via GPR41/43",
                "❌ TLR4 activation → NF-κB → TNF-α/IL-6 → IRS-1 inhibition → insulin resistance",
                "❌ Mitochondrial dysfunction; mucosal damage; reduces tight junctions",
                "⚠️ FXR/TGR5 signalling; GLP-1 secretion (depends on composition)",
                "✅ AhR pathway; gut barrier protection; anti-inflammatory",
            ],
            "Direction in T2D": ["Decreased","Decreased","Variable","Increased","Increased","Dysregulated","Decreased"]
        }
        metabolite_df = pd.DataFrame(metabolite_data)
        st.dataframe(metabolite_df, use_container_width=True, hide_index=True, height=320)

        st.markdown("---")
        st.subheader("Evidence Base")
        st.caption("""
        **Key references:**  
        • Qin J et al. (2012) *Nature* 490:55–60 — Gut metagenome & T2D (n=345)  
        • Karlsson FH et al. (2013) *Nature* 498:99–103 — European women cohort  
        • Plovier H et al. (2017) *Nature Medicine* — Akkermansia & metabolic syndrome  
        • Canfora EE et al. (2019) *Nature Reviews Gastroenterology* — SCFAs & T2D  
        • Gurung M et al. (2020) *EBioMedicine* 51:102590 — Meta-analysis  
        • Cani PD et al. (2007) *Diabetes* — Endotoxemia & insulin resistance  
        """)
