<!-- Logo + Title -->
<p align="center">
  <img src="docs/imgs/logo_shapbpt.png" alt="ShapBPT logo" width="150">
</p>

<h1 align="center">ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees</h1>

<p align="center">
  <b>AAAI-2026 (40th AAAI Conference on Artificial Intelligence), Singapore</b><br>
</p>

---

## 🔍 Overview

This repository contains the **experiments, notebooks, and precomputed results** for the paper:

> **ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees**  
> A novel eXplainable AI (XAI) method for image feature attribution based on data-aware binary partition trees.

This repo provides:

- All **experiments E1–E7** from the paper  
- **Notebooks** for quick and full replication  
- **Precomputed results (PDFs & CSVs)**  
- Ready-to-run ShapBPT usage examples

> **Note:** The actual *ShapBPT library* is hosted separately:  
> https://github.com/amparore/shap_bpt

---

## 📚 Experiments Summary

| Name | Dataset | Model | Task | Model Path / Type | Time |
|:----:|:--------|:------|:------|:-------------------|:------|
| E1 | ImageNet-S50 | ResNet50 | Classification | Pretrained | 7h 50m |
| E2 | ImageNet-S50 | Ideal | Controlled IoU | Pretrained | 4h 9m |
| E3 | ImageNet-S50 | SwinViT | Classification | Pretrained | 20h 6m |
| E4 | MS-COCO | YOLO11s | Object detection | Pretrained - `notebooks/E4_MS_COCO/yolo11s.pt` | 11h 42m |
| E5 | CelebA | CNN | Face attributes | Pretrained - `notebooks/E5_CelebA/models/model.pth` | 6h 14m |
| E6 | MVTec | VAE-GAN | Anomaly Detection | Pretrained - `notebooks/E6_XAD/models/` | 2h 56m |
| E7 | ImageNet-S50 | ViT-Base16 | Classification | Pretrained | 14h 48m |
| E8 | ImageNet-S50 | -- | Human Interpretation | - | - |
---

## ⚙️ 1. Setup
### 1.1 Create environment

```bash
conda create -n env_shapbpt python==3.9.18
conda activate env_shapbpt
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 1.2 LaTeX (optional but recommended)
Ubuntu:

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

Windows: Install **MikTeX**.

---
### 1.3 Clone tests repo

```bash
git clone https://github.com/rashidrao-pk/shap_bpt_tests
cd shap_bpt_tests
```

### 1.4 Clone main ShapBPT package and install

```bash
git clone https://github.com/amparore/shap_bpt
```
**_Note_**: follow all instructions to install ShapBPT package.

#### Test installation:

```python
import shap_bpt
print(shap_bpt.__version__)
```

---

### 1.5 Required datasets

Download and place into:

```
notebooks/datasets/
```

| Exp | Dataset | Download |
|:---:|:--------|:---------|
| E1/E2/E3/E7 | ImageNet-S50 | https://github.com/LUSSeg/ImageNet-S |
| E4 | MS-COCO 2017 val | https://cocodataset.org/#download |
| E5 | CelebAMask-HQ | https://github.com/switchablenorms/CelebAMask-HQ |
| E6 | MVTec AD | https://www.mvtec.com/company/research/datasets/mvtec-ad |

---

## 🚀 2. Minimal Example (ShapBPT)

```python
import shap_bpt

explainer = shap_bpt.Explainer(
    f_masked,                   # black box model
    image_to_explain, 
    num_explained_classes=4,
    verbose=True
)
MAX_EVALS_BUDGET = 1000
shap_values_bpt = explainer.explain_instance(
    MAX_EVALS_BUDGET,           # budget for explanation
    method="BPT",               # partioning method
    batch_size=4,               # bacth size
    # max_weight=None   
)

shap_values_aa = explainer.explain_instance(
    MAX_EVALS_BUDGET,
    method="AA",
    verbose_plot=False,
    batch_size=4,
    # max_weight=None
)

# shap_bpt.plot_owen_values(
#     explainer,
#     [shap_values_aa, shap_values_bpt],
#     class_names,
#     names=["AxisAligned", "BPT"]
# )

```

```python
# PLOT FEATURE ATTRIBUTION --> AA
shap_bpt.plot_owen_values(explainer, shap_values_aa, class_names)
```


<center><img src="docs/aa_plot.svg"></center>

```python
# PLOT FEATURE ATTRIBUTION --> BPT
shap_bpt.plot_owen_values(explainer, shap_values_bpt, class_names)
```
<center><img src="docs/bpt_plot.svg"></center>


---

## 📊 3. Precomputed Results

| Exp | Dataset | Model | PDF | CSV |
|:---:|:--------|:------|:-----|:-----|
| E1 | ImageNet-S50 | ResNet50 | `PDF/HTML_E1_real_resnet_gray_combined.pdf` | `csv_exp_E1_ImageNet_resnet_real_gray_logits.csv` |
| E2 | ImageNet-S50 | Ideal | `PDF/HTML_E2_ideal_resnet_gray_combined.pdf` | `csv_exp_E2_ImageNet_resnet_ideal_gray_logits.csv` |
| E3 | ImageNet-S50 | SwinViT | `PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf` | `csv_exp_E3_ImageNet_swin_trans_vit_real_gray_logits.csv` |
| E4 | MS-COCO | YOLO11s | `PDF/HTML_E4_yolo11s_gray_Combined.pdf` | `csv_exp_E4_yolo11s_gray_9.csv` |
| E5 | CelebA | CNN | `PDF/HTML_E5_CelebA_gray_combined.pdf` | `csv_exp_E5_IoU_face_1000_14_gray_brownhairs.csv` |
| E6 | MVTec | VAE-GAN | `PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf` | `csv_exp_E6_testresults_hazelnut_9_BPT_new_eval.csv` |
| E7 | ImageNet-S50 | ViT-Base16 | `PDF/HTML_E7_ViT__combined_100.pdf` | `csv_exp_E7_ImageNet_vit_real_gray_logits.csv` |

---

## 🔁 4. Reproduce Paper Results

### 4.1 Quick (few minutes)

- Run:

```
notebooks/N1_Fig1_and_Fig3.ipynb
```

- For Figures 4 & 6:

```
notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb
```

---

### 4.2 Full replication (long)

Run:

```
notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb
```

This computes:

- full saliency maps  
- IoU metrics  
- CSV files used in the paper  

Generate HTML visualizations:

```
notebooks/E1_E2_E3_E7/additional_material/N3_Create_HTML_File.ipynb
```

Runtimes (approx):

- E1: 24h  
- E2: 16h  
- E3: 30h  

---

## 🖥️ Hardware Used

| Device | CPU | RAM | GPU |
|:------|:----|:----:|:-----|
| Santech XN2 | Intel i9 13th Gen | 16GB | RTX 4070 |
| MacBook Pro | Apple M1 | 16GB | M1 GPU |

---

## 📁 Repo Structure

```
├── notebooks
│   ├── E1_E2_E3_E7
│   │   ├── N1_Run_experiments.ipynb
│   │   ├── N2_DrawPlotFig4_Fig6_from_CSV.ipynb
│   │   ├── N3_Create_HTML_File.ipynb
│   ├── E4_MS_COCO
│   ├── E5_CelebA
│   ├── E6_XAD
│   └── utils
├── PDF
├── results_logits_with_auc_clipped
├── saved_explanations
└── README.md
```

---

## 📑 Citation

```
@inproceedings{Rashid2026ShapBPT,
  title={ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees},
  author={Muhammad Rashid and Elvio G. Amparore and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## 👤 Author

**Muhammad Rashid**  
University of Turin × Rulex Labs × UniGranada  
LinkedIn: https://www.linkedin.com/in/rashid-rao-cuipakistan/

---

