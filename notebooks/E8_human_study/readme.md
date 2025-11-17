# Human Study Evaluation – ShapBPT vs Baseline XAI Methods  
### Statistical Analysis Notebook

This folder contains the analysis notebook used to evaluate **human-study results** for the paper:

> **_ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees_**

The notebook processes human responses collected for multiple XAI methods and performs a full statistical comparison across methods.

---

# 📌 Overview

This notebook analyzes a human-study experiment in which participants ranked saliency maps produced by several XAI methods:

- **BPT** (ShapBPT)
- **AA** (Axis-Aligned SHAP)
- **LIME**
- **GradCAM**

Participants were shown images, each corresponding to a *hidden method*, and asked to rank explanations from best (rank-1) to worst.

The notebook performs:

✔ **Data loading & preprocessing**  
✔ **Mapping survey questions to true underlying methods**  
✔ **Rank aggregation per subject**  
✔ **Average rank computation**  
✔ **Friedman statistical test (non-parametric ANOVA)**  
✔ **Nemenyi post-hoc test** via `scikit-posthocs`  
✔ **Rank-distribution plots**  
✔ **Method comparison tables**

---
# Explanations used for Human-Study:
Following four examples along with 4 XAI methods are used to collect the user preference data.
<img src='HumanStudy.svg'>
<caption> Explanations used to assess Human preferences </caption>

# 📂 Files Used

### **1. HumanStudyResults.xlsx**
This Excel file must be placed in the same folder as the notebook.  
It contains columns such as:

- `Subject`
- `QuestionID`
- `Option1`, `Option2`, …
- `Rank1`, `Rank2`, `Rank3`, `Rank4`
- `HiddenMethod` (or mapped in notebook)

### **2. Notebook**
```
HumanStudy_Analysis.ipynb
```
(Your notebook file.)

---

# 🧪 What the Notebook Computes

### **1. Question → Method Mapping**
Survey questions correspond to underlying saliency methods.  
The notebook defines:

```python
mapping_question_to_hiddenmethod = {
   1: "BPT",
   2: "AA",
   3: "LIME",
   4: "GradCAM",
   ...
}
```

This allows the analysis to know which method produced each explanation.

---

### **2. Rank Extraction & Long-Format Conversion**
The notebook converts Excel columns into a *long-format* ranking table:

| Subject | Question | Method | Rank |
|--------|----------|--------|------|

This is the format required for statistical tests.

---

### **3. Average Rank per Method**
The notebook computes:

```python
avg_rank = long.groupby("Method")["Rank"].mean()
```

and produces a bar-plot with LaTeX rendering.

---

### **4. Friedman Test (Statistical Significance)**  
Used to check if participants significantly preferred one method over others.

```python
friedmanchisquare(pivot['BPT'], pivot['GradCAM'], pivot['LIME'], pivot['AA'])
```

Outputs:
- χ² statistic  
- p-value  
- significance decision  

---

### **5. Nemenyi Post-Hoc Test**
If Friedman test is significant, the notebook performs:

```python
sp.posthoc_nemenyi_friedm(long_pivot_table)
```

This produces:
- Pairwise significance matrix  
- Which methods differ significantly  

Saved or printed as a colored table.

---

### **6. Rank Distribution Plots**
For each method, the notebook can generate:

- Histogram of rank-1, rank-2, rank-3, rank-4 counts  
- Or combined plots  

These help visualize participant preference patterns.

---

# ▶️ Running the Notebook

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-posthocs scipy openpyxl
```

Then open the notebook and run all cells:

```
HumanStudy_Analysis.ipynb
```

Ensure the Excel file is present:

```
HumanStudyResults.xlsx
```

---

# 📊 Outputs

The notebook produces:

### **1. Average Rank Table**
Example:

| Method   | Avg Rank |
|----------|----------|
| BPT      | 1.45     |
| AA       | 2.31     |
| LIME     | 2.87     |
| GradCAM  | 3.38     |

### **2. Friedman Test Result**
Example:

```
Friedman χ² = 41.2  
p-value = 2.1e-09  
→ Significant differences between methods.
```

### **3. Nemenyi Post-Hoc Matrix**
A table showing which method pairs differ significantly.

### **4. Rank Distribution Plots**
Barplots / histograms for each method.

---

# 📞 Contact
For questions or reproducing results:  
**Muhammad Rashid**  
University of Turin · Rulex Labs

---

This notebook forms the statistical backbone for the human evaluation section of the ShapBPT study.

