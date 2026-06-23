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