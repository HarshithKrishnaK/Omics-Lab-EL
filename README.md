# 🧬 Gut Microbiome Analysis Dashboard
### A Clinical Decision-Support Pipeline for Early-Stage Type 2 Diabetes (T2D) Risk Prediction & Therapeutic Simulation

The **Gut Microbiome Analysis Dashboard** is a computational biology tool designed to transform raw microbial relative abundance profiles into actionable clinical insights[cite: 1]. By treating the gut microbiome as a dynamic ecological network, this platform extracts key biomarkers to identify metabolic dysbiosis and flag T2D risk before traditional systemic indicators (like HbA1c or fasting blood glucose) alter significantly[cite: 1].

---

## 🚀 Key Features

*   **Dynamic File Ingestion:** Seamlessly parses local Tab-Separated Values (`.tsv`) clinical sequencing datasets into a live memory data frame[cite: 1].
*   **5-Biomarker Ecological Engine:** Quantifies community stability through five core metrics: Shannon Diversity, Firmicutes-to-Bacteroidetes (F:B) Ratio, Butyrate Producer Ratio, Pathobiont Load, and Species Richness[cite: 1].
*   **Multi-Tier Clinical Benchmarking:** Allows clinicians to toggle between absolute raw values, Healthy Reference Control Baselines, and active Type 2 Diabetes Cohort Profiles[cite: 1].
*   **Predictive Pathogenesis Cascade:** Maps patient risk scores across a live 6-step physiological cascade tracing the path from gut ecosystem collapse to clinical hyperglycemia[cite: 1].
*   **Digital Intervention Simulator:** A therapeutic sandbox letting doctors test prebiotics (Inulin/FOS), next-generation probiotics (*Akkermansia muciniphila*), or broad-spectrum antibiotic shocks to preview community shifting before issuing a prescription[cite: 1].

---

## 📊 The 5-Biomarker Engine Metrics

| Biomarker | Healthy Target | Clinical Rationale[cite: 1] |
| :--- | :---: | :--- |
| **Shannon Diversity Index** | > 3.8[cite: 1] | Reflects ecosystem resilience (The Rainforest Analogy)[cite: 1]. |
| **F:B Ratio** | 1.1[cite: 1] | Dictates global energy-harvesting and metabolic baseline profiles[cite: 1]. |
| **Butyrate Producer Ratio** | > 0.320[cite: 1] | Quantifies short-chain fatty acid capacity to preserve intestinal barrier wall tight junctions[cite: 1]. |
| **Pathobiont Load** | 0.000[cite: 1] | Gauges opportunistic inflammatory strains driving systemic endotoxemia[cite: 1]. |
| **Species Richness** | > 30[cite: 1] | Total headcount of unique functional bacterial taxa[cite: 1]. |

---

## 💻 Tech Stack

*   **Frontend & Dashboard Framework:** Streamlit
*   **Data Processing Pipeline:** Pandas, NumPy
*   **Interactive Analytics & Visualizations:** Plotly Express, Plotly Graph Objects

---

## 📦 Installation & Local Setup

To set up and run this dashboard locally, ensure you have Python installed, then execute the following commands in your terminal:
### 1. Clone the Repository
```bash
git clone [https://github.com/HarshithKrishnaK/Omics-Lab-EL.git](https://github.com/HarshithKrishnaK/Omics-Lab-EL.git)
cd Omics-Lab-EL

## 🚀 How to Run the Application

### 1. Clone the Repository

```bash
git clone https://github.com/HarshithKrishnaK/Omics-Lab-EL.git
cd Omics-Lab-EL
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard

```bash
streamlit run app_updated.py
```

### 5. Open the Dashboard

After the server starts, Streamlit will automatically open the application in your browser.

If it doesn't open automatically, visit:

```text
http://localhost:8501
```

---

## 📖 How to Use

### Step 1: Upload a Microbiome Dataset

* Use the file uploader in the sidebar.
* Upload a `.tsv` file containing:

  * `Taxon`
  * `Relative_Abundance`

You can also test the dashboard using the sample file:

```text
india_species_abundance_clean.tsv
```

### Step 2: Analyze Microbiome Health

The dashboard automatically calculates:

* Shannon Diversity Index
* Firmicutes:Bacteroidetes Ratio
* Butyrate Producer Ratio
* Pathobiont Load
* Species Richness

### Step 3: Review Clinical Risk Assessment

* Compare results with Healthy and T2D reference cohorts.
* View microbiome health status and dysbiosis indicators.
* Explore the T2D pathogenesis cascade.

### Step 4: Simulate Therapeutic Interventions

Use the intervention simulator to test:

* Prebiotics (Inulin/FOS)
* Probiotics (*Akkermansia muciniphila*)
* Antibiotic perturbations

Observe projected changes in microbial composition and health metrics in real time.

### Step 5: Interpret Results

The dashboard provides:

* Ecological health metrics
* Dysbiosis assessment
* T2D risk indicators
* Interactive visualizations
* Predicted intervention outcomes
