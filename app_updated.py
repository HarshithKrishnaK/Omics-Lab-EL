import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================
# 1. PAGE CONFIGURATION (MUST BE ABSOLUTE FIRST)
# ============================================================
st.set_page_config(
    page_title="Gut Microbiome Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Safe Custom CSS
# -------------------------------
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .risk-box {
        padding: 20px; border-radius: 12px; text-align: center;
        margin: 10px 0; border: 2px solid; background-color: #fff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Gut Microbiome Analysis Dashboard")
st.markdown("### Dynamic Nutrient Bioavailability Prediction Model")

# ============================================================
# 2. DATA LOADING CORE ENGINE
# ============================================================
@st.cache_data(show_spinner="Synchronizing multi-omics databases...")
def load_all_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "india_species_abundance_clean.tsv")
    
    # Load Local OTU Matrix Abundance
    try:
        otu_df = pd.read_csv(path, sep="\t", index_col=0)
        otu_df = otu_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    except Exception:
        # Fallback matrix if the file isn't found or empty
        otu_df = pd.DataFrame(
            index=["lactobacillus_acidophilus", "bifidobacterium_longum", "escherichia_coli", "faecalibacterium_prausnitzii"], 
            columns=["Sample1", "Sample2", "Sample3"]
        )
        otu_df.fillna(0.25, inplace=True)

    # Local Mock Clinical Data
    metadata_df = pd.DataFrame({
        "Sample_ID": otu_df.columns,
        "Disease_Status": np.random.choice(["Healthy", "T2D"], size=otu_df.shape[1]),
        "Fasting_Blood_Glucose": np.random.uniform(80, 180, size=otu_df.shape[1]),
        "HbA1c": np.random.uniform(5.2, 8.5, size=otu_df.shape[1])
    })

    return otu_df, metadata_df

otu, clinical_metadata = load_all_data()

# ============================================================
# 3. INTERACTIVE PATIENT FILE UPLOADER (DEDUPLICATED & RE-ORDERED)
# ============================================================
st.sidebar.markdown("### 📥 Patient Sample Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload a patient microbiome profile:", 
    type=["tsv", "csv"]
)

if uploaded_file is not None:
    # If a file is uploaded, process it into a clean Series
    sep = '\t' if uploaded_file.name.endswith('.tsv') else ','
    uploaded_df = pd.read_csv(uploaded_file, sep=sep)
    
    # Process table format (Species column + Abundance column) to clean Series mapping
    if 'Species' in uploaded_df.columns:
        baseline_abundance = uploaded_df.set_index('Species').iloc[:, 0]
    else:
        baseline_abundance = uploaded_df.set_index(uploaded_df.columns[0]).iloc[:, 0]
        
    st.sidebar.success(f"📋 Loaded: {uploaded_file.name}")
else:
    # Fallback to the matrix fallback calculation if no user file is present
    baseline_abundance = otu.mean(axis=1)
    st.sidebar.info("💡 Showing default baseline reference group.")

# Ensure values are float series for downstream mathematical functions
baseline_abundance = pd.to_numeric(baseline_abundance, errors='coerce').fillna(0.0)

# ============================================================
# 4. TAXONOMIC MAPPING & TRAIT DATA DICTIONARIES
# ============================================================
def get_genus(species):
    if isinstance(species, str):
        return species.split("_")[0].lower()
    return "unknown"

species_list = baseline_abundance.index.tolist()
genus_map = {sp: get_genus(sp) for sp in species_list}

genus_traits = {
    "lactobacillus":  {"SCFA":0.9, "pH_reduction":0.9, "Barrier_support":0.8, "Vitamin_Biosynthesis":0.6, "Siderophore":0.0},
    "bifidobacterium":{"SCFA":0.8, "pH_reduction":0.7, "Barrier_support":0.8, "Vitamin_Biosynthesis":0.7, "Siderophore":0.0},
    "faecalibacterium":{"SCFA":0.95,"pH_reduction":0.9,"Barrier_support":0.9, "Vitamin_Biosynthesis":0.4, "Siderophore":0.0},
    "roseburia":      {"SCFA":0.9, "pH_reduction":0.8, "Barrier_support":0.7, "Vitamin_Biosynthesis":0.2, "Siderophore":0.0},
    "akkermansia":    {"SCFA":0.3, "pH_reduction":0.3, "Barrier_support":0.95,"Vitamin_Biosynthesis":0.2, "Siderophore":0.0},
    "blautia":        {"SCFA":0.8, "pH_reduction":0.6, "Barrier_support":0.6, "Vitamin_Biosynthesis":0.3, "Siderophore":0.0},
    "bacteroides":    {"SCFA":0.6, "pH_reduction":0.3, "Barrier_support":0.5, "Vitamin_Biosynthesis":0.5, "Siderophore":0.1},
    "prevotella":     {"SCFA":0.7, "pH_reduction":0.4, "Barrier_support":0.6, "Vitamin_Biosynthesis":0.3, "Siderophore":0.0},
    "streptococcus":  {"SCFA":0.3, "pH_reduction":0.5, "Barrier_support":0.3, "Vitamin_Biosynthesis":0.3, "Siderophore":0.0},
    "clostridium":    {"SCFA":0.8, "pH_reduction":0.6, "Barrier_support":0.4, "Vitamin_Biosynthesis":0.2, "Siderophore":0.0},
    "ruminococcus":   {"SCFA":0.85,"pH_reduction":0.7, "Barrier_support":0.6, "Vitamin_Biosynthesis":0.3, "Siderophore":0.0},
    "escherichia":    {"SCFA":0.1, "pH_reduction":0.0, "Barrier_support":0.1, "Vitamin_Biosynthesis":0.1, "Siderophore":0.9},
    "enterobacter":   {"SCFA":0.1, "pH_reduction":0.0, "Barrier_support":0.1, "Vitamin_Biosynthesis":0.1, "Siderophore":0.8},
    "desulfovibrio":  {"SCFA":0.0, "pH_reduction":0.1, "Barrier_support":0.05,"Vitamin_Biosynthesis":0.0, "Siderophore":0.3},
}

default_trait = {"SCFA":0.2,"pH_reduction":0.1,"Barrier_support":0.1,"Vitamin_Biosynthesis":0.1,"Siderophore":0.2}
all_traits = []
for sp in species_list:
    genus = genus_map.get(sp, None)
    row = genus_traits.get(genus, default_trait).copy()
    row["Species"] = sp
    all_traits.append(row)
traits = pd.DataFrame(all_traits).set_index("Species")

# Nutrient Parameters
nutrients = {
    "Iron":        {"SCFA":0.4,"pH_reduction":0.3,"Barrier_support":0.2,"Siderophore":-0.6},
    "Vitamin_B12": {"Vitamin_Biosynthesis":0.7,"Barrier_support":0.2},
    "Folate":      {"Vitamin_Biosynthesis":0.6},
    "Calcium":     {"SCFA":0.5,"pH_reduction":0.4},
    "Magnesium":   {"SCFA":0.4},
    "Zinc":        {"Barrier_support":0.4,"Siderophore":-0.5}
}

def absorption_score(abundance, nutrient, normalize=True):
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

def simulate_addition(abundance, microbe, delta):
    new = abundance.copy()
    if microbe in new:
        new[microbe] = max(0.0, new[microbe] + delta)
    else:
        new[microbe] = max(0.0, delta)
    return new

# Dynamic Modeling Logic Lookups
T2D_REFERENCE = {
    "Shannon Diversity":       (3.8,  2.9,  True),
    "F:B Ratio":               (1.1,  2.4,  False),
    "Butyrate Producer Ratio": (0.32, 0.12, True),
    "Pathobiont Load":         (0.05, 0.18, False),
    "Species Richness":        (120,  78,   True),
}

glucose_response_weights = {
    "escherichia": +0.85, "streptococcus": +0.60, "clostridium": +0.40,
    "bifidobacterium": -0.40, "lactobacillus": -0.20, "roseburia": -0.60, "faecalibacterium": -0.80, "akkermansia": -0.70
}

INTERVENTION_BUNDLES = {
    "Inulin / FOS (prebiotic fibre)": {
        "description": "Fermentable fibre selectively feeds butyrate producers and Bifidobacterium.",
        "species_deltas": {"Faecalibacterium_prausnitzii": +0.03, "Bifidobacterium_longum": +0.02, "Escherichia_coli": -0.02},
        "glp1_boost": +0.15, "lps_reduction": -0.20
    },
    "Akkermansia muciniphila supplementation": {
        "description": "Improves insulin sensitivity and strengthens gut barrier networks.",
        "species_deltas": {"Akkermansia_muciniphila": +0.05, "Faecalibacterium_prausnitzii": +0.02},
        "glp1_boost": +0.12, "lps_reduction": -0.25
    },
    "Broad-Spectrum Antibiotic Protocol": {
        "description": "🚨 CRITICAL CLEARANCE: Aggressively decimates blooming pathobionts, but inflicts severe collateral damage on protective keystone taxa.",
        "species_deltas": {
            "Escherichia_coli": -0.15, 
            "Klebsiella_pneumoniae": -0.05, 
            "Ruminococcus_gnavus": -0.02,
            "Bifidobacterium_longum": -0.08, 
            "Faecalibacterium_prausnitzii": -0.05,
            "Lactobacillus_acidophilus": -0.03
        },
        "glp1_boost": -0.10, "lps_reduction": +0.05
    }
}
# ============================================================
# 5. MATHEMATICAL CALCULATION FUNCTIONS
# ============================================================
def compute_community_metrics(abundance_series):
    ab = abundance_series[abundance_series > 0]
    p = ab / ab.sum() if ab.sum() > 0 else ab
    shannon = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0
    
    firm_genera = {"lactobacillus","clostridium","ruminococcus","roseburia","faecalibacterium","streptococcus"}
    bact_genera = {"bacteroides","prevotella"}
    
    firm_ab = sum(v for sp,v in abundance_series.items() if genus_map.get(sp,"") in firm_genera)
    bact_ab = sum(v for sp,v in abundance_series.items() if genus_map.get(sp,"") in bact_genera)
    fb_ratio = firm_ab / bact_ab if bact_ab > 0 else 1.0
    
    butyrate_genera = {"faecalibacterium","roseburia","blautia"}
    pathobiont_genera = {"escherichia","clostridium","desulfovibrio"}
    
    butyrate_ab = sum(v for sp,v in abundance_series.items() if genus_map.get(sp,"") in butyrate_genera)
    pathobiont_ab = sum(v for sp,v in abundance_series.items() if genus_map.get(sp,"") in pathobiont_genera)
    total = abundance_series.sum()
    
    return {
        "Shannon Diversity":       round(shannon, 3),
        "F:B Ratio":               round(fb_ratio, 3),
        "Butyrate Producer Ratio": round(butyrate_ab / total if total > 0 else 0, 4),
        "Pathobiont Load":         round(pathobiont_ab / total if total > 0 else 0, 4),
        "Species Richness":        int((abundance_series > 0).sum()),
    }

def compute_dysbiosis_risk_score(metrics):
    weights = {"Butyrate Producer Ratio": 0.30, "Shannon Diversity": 0.25, "Pathobiont Load": 0.25, "F:B Ratio": 0.15, "Species Richness": 0.05}
    component_scores = {}
    for metric, (healthy_val, t2d_val, higher_is_healthier) in T2D_REFERENCE.items():
        raw = metrics[metric]
        if higher_is_healthier:
            score = 1.0 - (raw - t2d_val) / max(healthy_val - t2d_val, 1e-6)
        else:
            score = (raw - healthy_val) / max(t2d_val - healthy_val, 1e-6)
        component_scores[metric] = float(np.clip(score, 0.0, 1.0))
    total = sum(component_scores[m] * weights[m] for m in weights if m in component_scores)
    return round(total * 100, 1), component_scores

def apply_glucose_effect(abundance_series, glucose_factor):
    new_ab = abundance_series.copy()
    for sp in new_ab.index:
        g = genus_map.get(sp, "unknown")
        weight = glucose_response_weights.get(g, 0.0)
        new_ab[sp] = max(0.0, new_ab[sp] + (weight * glucose_factor * float(new_ab[sp]) * 1.5))
    total = new_ab.sum()
    return new_ab / total if total > 0 else new_ab

# ============================================================
# 6. SIDEBAR NAVIGATION CONTROLLER
# ============================================================
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Overview", "OTU Analysis", "Trait Coverage", "Nutrient Absorption", "Simulations",
        "📊 Dysbiosis Risk Score",        
        "🔗 Microbial Ecology & T2D",     
        "🔄 Feedback Loop Simulator",     
        "💊 Intervention Simulator"       
    ]
)

# ============================================================
# 7. TAB ROUTING RENDERING
# ============================================================
if page == "Overview":
    st.header("🧬 Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("🦠 Species Tracked", len(baseline_abundance))
    c2.metric("📊 Matrix Profiles", otu.shape[1])
    c3.metric("🔬 Target Micronutrients", len(nutrients))
    
    st.subheader("🌟 Population Abundance Top Profiles")
    top_species = baseline_abundance.sort_values(ascending=False).head(10)
    fig = px.bar(x=top_species.values, y=top_species.index, orientation='h', labels={"x":"Relative Abundance","y":"Taxon"}, color=top_species.values, color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

elif page == "OTU Analysis":
    st.header("OTU Abundance Analysis")
    fig = px.histogram(baseline_abundance, nbins=30, title="Taxonomic Distribution Profiles", labels={"value":"Relative Abundance"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Taxonomy Data Ledger")
    st.dataframe(pd.DataFrame({"Mean Abundance": baseline_abundance, "Assigned Genus": [genus_map.get(s) for s in baseline_abundance.index]}), use_container_width=True)

elif page == "Trait Coverage":
    st.header("Functional Trait Expression Map")
    fig = px.imshow(traits.T, color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Nutrient Absorption":
    st.header("Nutrient Bioavailability Modelling")
    scores = {n: absorption_score(baseline_abundance, n) for n in nutrients}
    fig = px.bar(x=list(scores.keys()), y=list(scores.values()), labels={"x":"Nutrient","y":"Calculated Value"}, color=list(scores.values()), color_continuous_scale="Plasma")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Simulations":
    st.header("Microbiome Perturbation Simulator")
    selected_bacteria = st.selectbox("Select Target Microbe:", sorted(list(traits.index)))
    selected_nutrient = st.selectbox("Select Target Nutrient:", list(nutrients.keys()))
    delta = st.slider("Abundance Alteration Vector (Δ):", -0.10, 0.30, 0.05, 0.01)
    
    b_score = absorption_score(baseline_abundance, selected_nutrient)
    mod_abundance = simulate_addition(baseline_abundance.to_dict(), selected_bacteria, delta)
    m_score = absorption_score(mod_abundance, selected_nutrient)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Matrix Score", f"{b_score:.4f}")
    m2.metric("Post-Perturbation Score", f"{m_score:.4f}")
    m3.metric("Net Shift Vector", f"{m_score - b_score:+.4f}", delta=m_score - b_score)

elif page == "🔗 Microbial Ecology & T2D":
    st.header("🧬 Live Disease Progression Tracker")
    st.markdown("This tracker maps your raw microbiome data against the classic multi-step pathogenesis cascade for Type 2 Diabetes.")
    
    metrics = compute_community_metrics(baseline_abundance)
    dysbiosis_score, _ = compute_dysbiosis_risk_score(metrics)
    
    shannon = metrics.get('Shannon Diversity', 2.0)
    butyrate_ratio = metrics.get('Butyrate Producer Ratio', 0.3)
    pathobiont_load = metrics.get('Pathobiont Load', 0.0)
    
    # Threshold Evaluators
    step1_fail = shannon < 2.5 or metrics.get('Species Richness', 15) < 15
    step2_fail = butyrate_ratio < 0.25 or pathobiont_load > 0.05
    step3_fail = butyrate_ratio < 0.20
    step4_fail = pathobiont_load > 0.10
    step5_fail = dysbiosis_score > 50.0
    step6_fail = dysbiosis_score > 70.0

    def render_step(step_num, title, description, is_failed):
        if is_failed:
            st.error(f"🔴 **Step {step_num}: {title}**\n\n*{description}*")
        else:
            st.success(f"🟢 **Step {step_num}: {title}**\n\n*Ecosystem Status: Stable. No active marker detected.*")

    st.subheader("Current Pathogenesis Status")
    
    render_step(1, "Ecology — Community Imbalance", "CRITICAL: Keystone butyrate producers decline. Gram-negative opportunists bloom.", step1_fail)
    render_step(2, "Microbial Metabolism — Metabolic Output Shifts", "WARNING: SCFA production falls. LPS and systemic endotoxins increase.", step2_fail)
    render_step(3, "Gut Physiology — Barrier Integrity Fails", "DAMAGE DETECTED: Butyrate deficit leads to increased intestinal permeability ('leaky gut').", step3_fail)
    render_step(4, "Systemic — Endotoxemia & Inflammation", "ALERT: LPS enters portal circulation, triggering chronic inflammatory cytokines.", step4_fail)
    render_step(5, "Metabolic — Insulin Signalling Disrupted", "METABOLIC BLOCK: Cytokines block insulin receptor substrates, driving insulin resistance.", step5_fail)
    render_step(6, "Clinical — Glucose Dysregulation", "CLINICAL STAGE: Hepatic glucose output goes uncontrolled. HbA1c rises progressively.", step6_fail)

    st.markdown("---")
    if step5_fail:
        st.error("🚨 **Clinical Assessment:** High-risk metabolic disruption in progress. Structural ecosystem restoration needed to resensitize insulin pathways.")
    elif step3_fail:
        st.warning("⚠️ **Clinical Assessment:** The patient is demonstrating early-stage gut barrier degradation. Immediate prebiotic intervention recommended to halt systemic progression.")
    else:
        st.success("🎯 **Clinical Assessment:** Homeostasis maintained. Ecosystem diversity is actively protective against metabolic inflammation.")

elif page == "📊 Dysbiosis Risk Score":
    st.header("📊 T2D Dysbiosis Risk Assessment")
    st.markdown("Analyze baseline ecological biomarkers and compare patient metrics against standardized clinical reference cohorts.")

    metrics = compute_community_metrics(baseline_abundance)
    dysbiosis_score, _ = compute_dysbiosis_risk_score(metrics)
    
    shannon = metrics.get('Shannon Diversity', 1.777)
    fb_ratio = metrics.get('F:B Ratio', 0.434)
    butyrate_ratio = metrics.get('Butyrate Producer Ratio', 0.208)
    pathobiont_load = metrics.get('Pathobiont Load', 0.000)
    richness = metrics.get('Species Richness', 13.0)

    st.subheader("Overall Microbiome Dysbiosis Index")
    if dysbiosis_score < 33:
        st.success(f"**Current Status:** 🟢 Low Risk (Homeostasis)")
    elif 33 <= dysbiosis_score <= 66:
        st.warning(f"**Current Status:** 🟡 Moderate Risk (Early Dysbiosis)")
    else:
        st.error(f"**Current Status:** 🔴 High Risk (Severe Metabolic Dysbiosis)")
        
    st.progress(int(min(max(dysbiosis_score, 0.0), 100.0)))
    st.markdown(f"🔬 **Calculated Risk Matrix Score:** `{dysbiosis_score:.1f} / 100`")

    st.markdown("---")
    st.subheader("👥 Clinical Population Benchmarking")
    cohort_tab = st.radio(
        "Select Comparison Cohort:",
        ["None (Raw Numbers)", "Healthy Reference Control Group", "Type 2 Diabetes Cohort Profile"],
        horizontal=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    
    if cohort_tab == "None (Raw Numbers)":
        col1.metric("Shannon Diversity", f"{shannon:.3f}")
        col2.metric("F:B Ratio", f"{fb_ratio:.3f}")
        col3.metric("Butyrate Ratio", f"{butyrate_ratio:.3f}")
        col4.metric("Pathobiont Load", f"{pathobiont_load:.3f}")
        col5.metric("Species Richness", f"{int(richness)}")
        
    elif cohort_tab == "Healthy Reference Control Group":
        col1.metric("Shannon Diversity", f"{shannon:.3f}", delta=f"{shannon - 3.8:.3f} (Target: >3.8)", delta_color="inverse" if shannon < 3.8 else "normal")
        col2.metric("F:B Ratio", f"{fb_ratio - 1.1:+.3f} vs Target 1.1")
        col3.metric("Butyrate Ratio", f"{butyrate_ratio:.3f}", delta=f"{butyrate_ratio - 0.32:.3f} (Target: >0.32)", delta_color="inverse" if butyrate_ratio < 0.32 else "normal")
        col4.metric("Pathobiont Load", f"{pathobiont_load:.3f}", delta=f"{pathobiont_load - 0.05:.3f} (Target: <0.05)", delta_color="inverse" if pathobiont_load > 0.05 else "normal")
        col5.metric("Species Richness", f"{int(richness)}", delta=f"{int(richness - 120)} (Target: >120)", delta_color="inverse" if richness < 120 else "normal")

    elif cohort_tab == "Type 2 Diabetes Cohort Profile":
        col1.metric("Shannon Diversity", f"{shannon:.3f}", delta=f"{shannon - 2.9:.3f} (T2D Avg: 2.9)")
        col2.metric("F:B Ratio", f"{fb_ratio:.3f}", delta=f"{fb_ratio - 2.4:.3f} (T2D Avg: 2.4)")
        col3.metric("Butyrate Ratio", f"{butyrate_ratio:.3f}", delta=f"{butyrate_ratio - 0.12:.3f} (T2D Avg: 0.12)")
        col4.metric("Pathobiont Load", f"{pathobiont_load:.3f}", delta=f"{pathobiont_load - 0.18:.3f} (T2D Avg: 0.18)")
        col5.metric("Species Richness", f"{int(richness)}", delta=f"{int(richness - 78)} (T2D Avg: 78)")

    st.markdown("---")
    st.subheader("🔍 Biomarker Explanations & Clinical Ranges")
    with st.expander("📊 Shannon Diversity Index"):
        st.markdown(f"* **Your Value:** `{shannon:.3f}`\n* **Healthy Target:** > 3.5\n* Measures structural alpha-diversity.")
    with st.expander("⚖️ Firmicutes-to-Bacteroidetes (F:B) Ratio"):
        st.markdown(f"* **Your Value:** `{fb_ratio:.3f}`\n* Structural ratio between major phyla.")
    with st.expander("🧪 Butyrate Producer Ratio"):
        st.markdown(f"* **Your Value:** `{butyrate_ratio:.3f}`\n* Percentage of SCFA synthesizing populations.")

elif page == "🔄 Feedback Loop Simulator":
    st.header("🔄 Real-Time Patient Glucose Tracker")
    st.markdown("Type in your actual blood sugar reading to see how your current metabolic state impacts your gut ecosystem.")
    
    patient_glucose = st.number_input("Enter your Fasting Blood Sugar (mg/dL):", min_value=70, max_value=300, value=100, step=5)
    
    if patient_glucose > 100:
        g_factor = min((patient_glucose - 100) / 200.0, 1.0)
    else:
        g_factor = 0.0

    if patient_glucose <= 100:
        st.success("🎯 Your blood sugar is in a healthy range! Helpful bacteria are thriving.")
    elif 100 < patient_glucose <= 125:
        st.warning("⚠️ Pre-diabetes Range: Elevated sugar levels are starting to introduce environmental stress.")
    else:
        st.error("🚨 Hyperglycemic Range: High blood sugar is causing toxic pressure on protective gut microbes.")

    if g_factor > 0:
        shifted = apply_glucose_effect(baseline_abundance, g_factor)
        b_risk, _ = compute_dysbiosis_risk_score(compute_community_metrics(baseline_abundance))
        s_risk, _ = compute_dysbiosis_risk_score(compute_community_metrics(shifted))
        
        st.markdown("---")
        st.subheader("Predictive Impact on Your Gut Health:")
        st.metric(label="Your Projected Dysbiosis Risk Score", value=f"{s_risk}/100", delta=f"+{s_risk - b_risk:.1f} Increase in Diabetes Strain", delta_color="inverse")

elif page == "💊 Intervention Simulator":
    st.header("Therapeutic Intervention Engine")
    protocol = st.selectbox("Select Intervention Framework:", list(INTERVENTION_BUNDLES.keys()))
    bundle = INTERVENTION_BUNDLES[protocol]
    st.write(bundle["description"])
    
    if st.button("Execute Biotic Simulation"):
        new_ab = baseline_abundance.copy()
        for partial, d in bundle["species_deltas"].items():
            matches = [m for m in new_ab.index if partial.lower() in m.lower()]
            for m in matches:
                new_ab[m] = max(0.0, new_ab[m] + d)
        new_ab = new_ab / new_ab.sum()
        
        b_risk, _ = compute_dysbiosis_risk_score(compute_community_metrics(baseline_abundance))
        n_risk, _ = compute_dysbiosis_risk_score(compute_community_metrics(new_ab))
        
        st.metric("New Community Dysbiosis Index", f"{n_risk}", delta=f"{n_risk - b_risk:.1f}", delta_color="inverse")