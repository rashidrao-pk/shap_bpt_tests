# ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees <img src="docs/imgs/logo_shapbpt.png" alt="ShapBPT logo" align="right" width="150">

<p align="center">
  <b>AAAI-2026 (40th AAAI Conference on Artificial Intelligence), Singapore</b><br>
</p>

<img src="https://img.shields.io/badge/version-v0.0.0-rc0" alt="Version">
      <a href ="https://github.com/rashidrao-pk/shap_bpt_tests/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
      </a> <a href="https://github.com/rashidrao-pk/">
<img src="https://img.shields.io/github/contributors/rashidrao-pk/shap_bpt_tests" alt="GitHub contributors">
</a> <img src="https://img.shields.io/github/repo-size/rashidrao-pk/shap_bpt_tests" alt="GitHub repo size">
      <a href="https://github.com/rashidrao-pk/">
        <img src="https://img.shields.io/github/commit-activity/t/rashidrao-pk/shap_bpt_tests" alt="GitHub commit activity (branch)">
      </a>
<img src="https://img.shields.io/github/last-commit/rashidrao-pk/shap_bpt_tests" alt="GitHub last commit">
<a href="https://github.com/rashidrao-pk/shap_bpt_tests/forks">
<img src="https://img.shields.io/github/forks/rashidrao-pk/shap_bpt_tests?style=flat" alt="GitHub forks">
</a>
<a href="https://github.com/rashidrao-pk/shap_bpt_tests/stargazers">
<img src="https://img.shields.io/github/stars/rashidrao-pk/shap_bpt_tests?style=flat" alt="GitHub Repo stars">
</a>
<img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2Frashidrao-pk&label=Visitors&countColor=%23263759&style=flat" alt="Visitors">

## 🎉 ShapBPT Experiments — v1.0 ([AAAI-26](https://aaai.org/conference/aaai/aaai-26/) Paper Release)

## Availability
ShapBPT Package: 
Technical Appendix: 

- Code — **_https://github.com/amparore/shap_bpt_**
- Tests — **_https://github.com/rashidrao-pk/shap_bpt_tests_**
- Tech. Appendix — **_https://zenodo.org/records/17570695_**

---
**Experiments repository for the paper**

> **ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees**  
> Accepted at **AAAI-26 (40th AAAI Conference on Artificial Intelligence), Singapore**  

---

## 🔍 What this repo is

This repository contains **all experiments, notebooks, and precomputed results** used in the ShapBPT paper, including:

- Experiments **E1–E8** (ImageNet-S50, MS-COCO, CelebA, MVTec AD, and human study)
- Jupyter notebooks for **quick reproduction of main figures**
- **Precomputed PDFs & CSVs** for tables and plots in the paper
- Minimal examples of how to use **ShapBPT explanations** in practice

> ⚠️ **Important:** This repo does **not** contain the ShapBPT library itself.  
> The core implementation lives here:  
> 🔗 https://github.com/amparore/shap_bpt

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
# Python deps
pip install -r requirements.txt
# PyTorch with CUDA (adjust for your system if needed)
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

💡 You do not need all datasets to run the quick demo notebooks.
Only download the ones for the experiments you want to reproduce.

---

## 🧩 2. Minimal ShapBPT Example

```python
import shap_bpt

explainer = shap_bpt.Explainer(
    f_masked,            # your black-box model f(x ⊙ m)
    image_to_explain,    # H×W×C numpy array or tensor (see shap_bpt docs)
    num_explained_classes=4,
    verbose=True
)

MAX_EVALS_BUDGET = 1000

# BPT partitioning
shap_values_bpt = explainer.explain_instance(
    MAX_EVALS_BUDGET,
    method="BPT",        # data-aware binary partition tree
    batch_size=4,
)

# Axis-aligned baseline
shap_values_aa = explainer.explain_instance(
    MAX_EVALS_BUDGET,
    method="AA",
    verbose_plot=False,
    batch_size=4,
)

# Plot attributions
shap_bpt.plot_owen_values(explainer, shap_values_aa, class_names)
shap_bpt.plot_owen_values(explainer, shap_values_bpt, class_names)

```

---

## 📊 3. Precomputed Results

| Exp | Dataset | Model | PDF | CSV |
|:---:|:--------|:------|:-----|:-----|
| E1 | ImageNet-S50 | ResNet50 | [_PDF/HTML_E1_real_resnet_gray_combined.pdf_](/PDF/HTML_E1_real_resnet_gray_combined.pdf) | _csv_exp_E1_ImageNet_resnet_real_gray_logits.csv_ |
| E2 | ImageNet-S50 | Ideal | [_PDF/HTML_E2_ideal_resnet_gray_combined.pdf_](/PDF/HTML_E2_ideal_resnet_gray_combined.pdf) | _csv_exp_E2_ImageNet_resnet_ideal_gray_logits.csv_ |
| E3 | ImageNet-S50 | SwinViT | [_PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf_](/PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf) | _csv_exp_E3_ImageNet_swin_trans_vit_real_gray_logits.csv_ |
| E4 | MS-COCO | YOLO11s | [_PDF/HTML_E4_yolo11s_gray_Combined.pdf_](/PDF/HTML_E4_yolo11s_gray_Combined.pdf) | _csv_exp_E4_yolo11s_gray_9.csv_ |
| E5 | CelebA | CNN | [_PDF/HTML_E5_CelebA_gray_combined.pdf_](/PDF/HTML_E5_CelebA_gray_combined.pdf) | _csv_exp_E5_IoU_face_1000_14_gray_brownhairs.csv_ |
| E6 | MVTec | VAE-GAN | [_PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf_](/PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf) | _csv_exp_E6_testresults_hazelnut_9_BPT_new_eval.csv_ |
| E7 | ImageNet-S50 | ViT-Base16 | [_PDF/HTML_E7_ViT__combined_100.pdf_](/PDF/HTML_E7_ViT__combined_100.pdf) | _csv_exp_E7_ImageNet_vit_real_gray_logits.csv_ |
---


## 4. Figure-to-Notebook Mapping

### ✅ 4.1 Main Paper

| Paper Figure | What it Shows                           | Notebook Path                                     |
| ------------ | --------------------------------------- | ------------------------------------------------- |
| **Fig. 1**   | Overview + comparison example           | [notebooks/N1_Fig1_and_Fig3.ipynb](/notebooks/N1_Fig1_and_Fig3.ipynb)                 |
| **Fig. 2**   | BPT partitioning visualization          | *Generated from core library demos*               |
| **Fig. 3**   | Qualitative examples (ResNet/Swin/ViT)  | [notebooks/N1_Fig1_and_Fig3.ipynb](/notebooks/N1_Fig1_and_Fig3.ipynb)                 |
| **Fig. 4**   | IoU comparison (AA vs BPT)              | [/notebooks/E1_E2_E3_E7/N1_DrawPlotFig4_Fig6_from_CSV.ipynb](//notebooks/E1_E2_E3_E7/N1_DrawPlotFig4_Fig6_from_CSV.ipynb) |
| **Fig. 5**   | Quantitative Analysis           | [/notebooks/N2_summary_plots.ipynb](//notebooks/N2_summary_plots.ipynb)                    |
---
### ✅ 4.2 Technical Appendix ([available here](https://zenodo.org/records/17570695))
---
| Paper Figure | What it Shows                           | Notebook Path                                     |
| ------------ | --------------------------------------- | ------------------------------------------------- |
| **Fig. 6,Fig. 9,Fig. 10, & Fig. 19**   | Extended heatmaps for Exp E1,E2,E2 & E7                    | [/notebooks/E1_E2_E3_E7/N1_1_Run_experiments_testing.ipynb](//notebooks/E1_E2_E3_E7/N1_1_Run_experiments_testing.ipynb) |
| **Fig. 7,Fig.8, Fig.11  & Fig. 20**   | Results on all images for Exp E1,E2,E2 & E7 | [/notebooks/E1_E2_E3_E7/N2_1_DrawPlot_Fig5_Fig6_Fig7_from_CSV.ipynb](//notebooks/E1_E2_E3_E7/N2_1_DrawPlot_Fig5_Fig6_Fig7_from_CSV.ipynb)                       |
| **Fig. 12**   | Extended heatmaps for Exp. E4 (Object Detection on MS COCO) | [/notebooks/E4_MS_COCO/N1_MS_COCO_testing.ipynb.ipynb](/notebooks/E4_MS_COCO/N1_MS_COCO_testing.ipynb.ipynb)                       |
| **Fig. 13**   | Results on all images | [E4_MS_COCO/N2_DrawPlot_from_CSV.ipynb](/notebooks/E4_MS_COCO/N2_DrawPlot_from_CSV.ipynb)                       |
| **Fig. 14**   | Extended heatmaps for Exp. E5 (Facial Attributes Recongnition on  CelebA dataset) | [/notebooks/E5_CelebA/N1_Run_experiments_CelebA.ipynb](/notebooks/E5_CelebA/N1_Run_experiments_CelebA.ipynb)                       |
| **Fig. 15**   | Results for Exp. E5 (Facial Attributes Recongnition on  CelebA dataset) | [/notebooks/E5_CelebA/N2_DrawPlot_from_CSV.ipynb](/notebooks/E5_CelebA/N2_DrawPlot_from_CSV.ipynb)                       |
| **Fig. 16**   | Framework for Anomaly Detection | using drawing tool                       |
| **Fig. 17**   | Extended heatmaps Anomaly detection explanations (using VAE-GAN on MVTec dataset)  | [/notebooks/E6_XAD/N1_XAD_HAZELNUT.ipynb](/notebooks/E6_XAD/N1_XAD_HAZELNUT.ipynb)                       |
| **Fig. 18**   | Results on all images  | [/notebooks/E6_XAD/N2_DrawPlot_from_CSV.ipynb](/notebooks/E6_XAD/N2_DrawPlot_from_CSV.ipynb)                       |
| **Fig. 21**   | Human study — Explanations  | -                   |
| **Fig. 22**   | Human study — ranking of explanations   | [/notebooks/E8_human_study/HumanStudyResults.ipynb](/notebooks/E8_human_study/HumanStudyResults.ipynb)                   |
| **Fig. 23**   | Results for Convergence Test | [/notebooks/E1_E2_E3_E7/N2_2_DrawPlot_Fig5_Fig6_Fig7_from_CSV.ipynb](/notebooks/E1_E2_E3_E7/N2_2_DrawPlot_Fig5_Fig6_Fig7_from_CSV.ipynb)                       |
| **Table 2**   | Annova Analysis   | [notebooks/N2_summary_plots.ipynb](/notebooks/N2_summary_plots.ipynb)                   |


## 🔁 5. Reproduce Paper Results

### 5.1 Quick (few minutes)

- Run:

```
notebooks/N1_Fig1_and_Fig3.ipynb
```

- For Figures 4 & 6:

```
notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb
```

---

### 5.2 Full replication (long)

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
### Keywords 🔍
Explainable AI · XAI · Computer Vision · Object Localization

## 🙏 Acknowledgments

We gratefully acknowledge the following contributions and resources that supported this work:

### 💠 Funding
This work received funding from the European Union’s Horizon research and innovation programme **Chips JU** under Grant Agreement No. **101139769**, as part of the [**DistriMuSe Project**](https://distrimuse.eu/) (HORIZON-KDT-JU-2023-2-RIA).  
The Joint Undertaking receives support from the European Union and the participating member states.

### 🧠 Models & Pretrained Weights
We thank the developers of the model architectures and pretrained weights used in our experiments, including:

- [**Swin Transformer (Microsoft)**](https://github.com/microsoft/Swin-Transformer)  
- [**Vision Transformer (PyTorch)**](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html?highlight=vit+b)  
- [**Facial attribute CNN by Kartik Batra**](https://www.kaggle.com/code/kartikbatra/multilabelclassification/notebook)  
- [**VAE-GAN model & weights**](https://github.com/rashidrao-pk/anomaly_detection_trust_case_study)  
- [**YOLO Models by Ultralytics**](https://docs.ultralytics.com/models/yolov11/)

### 🗄️ Datasets
We acknowledge the dataset curators whose work made this project possible:

- [**ImageNet**](https://www.image-net.org)  
- [**ImageNet-S<sub>50</sub>**](https://github.com/LUSSeg/ImageNet-S)  
- [**CelebA-HQ**](https://github.com/switchablenorms/CelebAMask-HQ)  
- [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)  
- [**MS-COCO**](https://cocodataset.org)

### 📘 Related Work Inspiration
Some notebook design ideas were inspired by the excellent documentation and examples in **Shap-IQ**:  
🔗 https://shapiq.readthedocs.io/en/latest/index.html  
We appreciate their contributions to interpretable machine learning research.



## Contributors
<a href="https://github.com/rashidrao-pk/shap_bpt_tests/graphs/contributors">
  <img src="http://contributors.nn.ci/api?repo=rashidrao-pk/shap_bpt_tests" alt="" />
</a>
<br>

> [!NOTE]
> Contributions to improve the completeness of this list are greatly appreciated. If you come across any overlooked papers, please **feel free to [*create pull requests*](https://github.com/rashidrao-pk/shap_bpt_tests/pulls), [*open issues*](https://github.com/rashidrao-pk/shap_bpt_tests/issues) or contact me via [*email*](mailto:muhammad.rashid@unito.it)**. Your participation is crucial to making this repository even better.




