# ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees

This repository contains all experiments and supplementary material for the paper:

> **_ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees_**  
> A novel eXplainable AI (**XAI**) method accepted at **AAAI-26**.

ShapBPT introduces a data-aware binary partitioning mechanism for computing Shapley-based image explanations more efficiently and with higher fidelity.  
This repository includes **complete experiment pipelines**, **pretrained models**, **notebooks**, **evaluation scripts**, and all **figures** required to reproduce results from the paper.

---

# 📚 Experiments Summary

| Exp | Dataset | Model | Task | Model Path / Type | Runtime |
|:---:|:--------|:------|:------|:------------------|:---------|
| **E1** | ImageNet-S50 | ResNet-50 | Image Classification | Pretrained | **7h 50m** |
| **E2** | ImageNet-S50 | Ideal Linear | Controlled IoU Benchmark | Pretrained | **4h 9m** |
| **E3** | ImageNet-S50 | Swin-ViT | Image Classification | Pretrained | **20h 6m** |
| **E4** | MS-COCO | YOLO11s | Object Detection | `notebooks/E4_MS_COCO/yolo11s.pt` | **11h 42m** |
| **E5** | CelebA | CNN | Facial Attribute Localization | `notebooks/E5_CelebA/models/model.pth` | **6h 14m** |
| **E6** | MVTec AD | VAE-GAN | Visual Anomaly Detection | `notebooks/E6_XAD/models/` | **2h 56m** |
| **E7** | ImageNet-S50 | ViT-Base16 | Image Classification | Pretrained | **14h 48m** |
| **E8** | ImageNet-S50 | — | Human Interpretation Study | — | — |

---

# 📁 Repository Structure

This repository contains all experimental setups for **ShapBPT** across multiple vision tasks.

### **1. `E1_to_E3/` – ImageNet-S50 (Image Localization)**
Experiments for:
- ResNet-50 (E1)
- Ideal Linear model (E2)
- SwinViT (E3)

Includes:
- Classifier wrappers  
- ShapBPT baseline comparisons  
- Heatmaps & IoU visualizations  
- CSV outputs for Figures 3–6  

---

### **2. `E4_MS_COCO/` – Object Detection (MS-COCO)**
Contains:
- YOLO11s pretrained model
- Full pipeline for object-detection explanations  
- Heatmaps, ground-truth overlays, IoU plots  
- Full validation set evaluation  
- Notebooks to reproduce Figures 9–10  

---

### **3. `E5_CelebA/` – Facial Attribute Localization**
Includes:
- Pretrained CNN for CelebA attributes
- Ablation analysis across background replacement settings  
- Boxplots (Figure 14)  
- HTML visualization pages for heatmaps and IoU results  
- Scripts to reproduce Figures 12–13  

---

### **4. `E6_XAD/` – Visual Anomaly Detection (MVTec AD)**
Contains:
- VAE-GAN black-box model (30000-epoch version)
- Explanation pipeline for anomaly detection  
- Pixel anomaly maps, reconstructions, ShapBPT masks  
- CSV + plots for anomaly-IoU (Figure 11–12)  

---

### **5. `dataset/`**
This folder should contain:
- ImageNet-S50  
- MS-COCO 2017 val  
- CelebAMask-HQ  
- MVTec AD dataset  

Follow instructions in each experiment folder for correct placement.

---

### **6. `paper_figures/`**
Contains all **sub-figures** used to assemble:
- **Figure 1**
- **Figure 2**

These are generated automatically from the `E1_to_E3`, `E4`, `E5`, and `E6` experiment folders.

---

### **7. `utils/`**
Shared utilities across experiments:
- Visualization helpers  
- Masking utilities  
- Plotting styles  
- Background replacement functions  
- Shared evaluation metrics  

---

# 🔍 Notes
- Each experiment folder contains its own detailed README explaining dataset setup, model requirements, and notebook usage.  
- All results are reproducible using the provided CSVs, HTML pages, and precomputed model weights.  
- GPU is recommended for E1–E7 due to evaluation budgets.

