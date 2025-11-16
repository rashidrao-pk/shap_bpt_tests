# ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees  
### Supplementary Material – Anomaly Detection (Experiment E6)

This directory contains the complete supplementary material for **Experiment E6** from the paper:

> **_ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees_**

Experiment E6 focuses on **eXplainable Anomaly Detection (XAD)** using a **VAE-GAN black-box model** trained on the **MVTec AD** dataset.  
The goal is to show how ShapBPT provides meaningful explanations of the anomaly-detection process by highlighting relevant defective regions.

This folder includes:
- Pretrained VAE-GAN model  
- Notebooks for explanations, IoU evaluation, HTML visualization  
- CSV results, boxplots, and complete heatmap/Iou sets  
- Full reproduction of all E6 results in the supplementary material  

---

# 🔧 Dependencies and Installation

### **Python Version**
- Python **3.9.18**

### **Required Libraries**
- TensorFlow (used for the VAE-GAN model)
- NumPy, OpenCV, matplotlib, tqdm
- GPU/CUDA recommended for faster evaluation

### **Environment Setup**
```bash
conda create -n ad python==3.9.18
conda activate ad
pip install -r requirements.txt
```

---

# 📂 Dataset

Experiment E6 uses the **MVTec AD** anomaly-detection dataset:

Dataset link:  
https://www.mvtec.com/company/research/datasets/mvtec-ad

Download the full dataset and place it inside:

```
notebooks/dataset/MVTec/
```

Each object category (e.g., hazelnut, bottle, tile) should maintain its original folder structure.

---

# 🧠 Model Setup (VAE-GAN)

A **VAE-GAN model** trained for **30000 epochs** on the **Hazelnut** category is bundled for this experiment.

Place the pretrained folder here:

```
notebooks/E6_XAD/models/hazelnut_VAE_GAN_30000/
```

### **Required files (must be exactly 11 files):**
- hazelnut_VAE_GAN_30000.csv  
- hazelnut_VAE_GAN_30000_decoder.data-00000-of-00001  
- hazelnut_VAE_GAN_30000_decoder.index  
- hazelnut_VAE_GAN_30000_discriminator.data-00000-of-00001  
- hazelnut_VAE_GAN_30000_discriminator.index  
- hazelnut_VAE_GAN_30000_encoder.data-00000-of-00001  
- hazelnut_VAE_GAN_30000_encoder.index  
- hazelnut_VAE_GAN_30000_model.data-00000-of-00001  
- hazelnut_VAE_GAN_30000_model.index  
- hazelnut_VAE_GAN_30000_vae.data-00000-of-00001  
- hazelnut_VAE_GAN_30000_vae.index  

You may retrain the model with more epochs for improved reconstruction, but for reproducibility the above version should be used.

---

# 📁 Folder Structure

```
XAD/
│
├── N1_XAD_HAZELNUT.ipynb
├── N2_DrawPlot_from_CSV.ipynb
├── N3_Create_HTMLs.ipynb
│
├── models/               # VAE-GAN pretrained model
├── results/              # All outputs
│   ├── csv/              # Metric CSVs
│   ├── boxplots/         # Precomputed paper boxplots
│   ├── hazelnut/         # Per-image explanations
│   ├── test_results/     # Full-test results
│   ├── imgs_hazelnut_heatmaps.html
│   ├── imgs_hazelnut_iou.html
```

---

# 📝 Notebooks and Their Functions

### **1. `models.py`**
Located at:
```
notebooks/XAD/main/models.py
```
Contains:
- VAE-GAN Encoder, Decoder, Discriminator  
- Reconstruction and sampling utilities  

---

### **2. `utils.py`**
Located at:
```
notebooks/XAD/utils.py
```
Contains helper functions used for:
- image preprocessing  
- anomaly map computation  
- mask and IoU evaluation  
- result saving (heatmaps, IoU maps)  

---

### **3. `N1_XAD_HAZELNUT.ipynb`** → Main experiment notebook

Supports:

#### ➤ **Single-example explanations**
Set:
```
plot_selected_results = True
```
(Used to produce supplementary **Figure 11**.)

Outputs saved in:
```
results/hazelnut/paper_figure/
```

#### ➤ **Full test set evaluation**
Enable in **Cell 55**:
```
run_full_set = True
```
This computes:
- heatmaps  
- IoU maps  
- pixel-level anomaly scores  
- CSV files  

Saved into:
```
results/test_results/
results/csv/testresults_hazelnut_30000_9_BPT.csv
```

---

### **4. `N2_DrawPlot_from_CSV.ipynb`**  
Generates boxplots (supplementary **Figure 12**):

Uses either:

- Precomputed CSV:
```
precomputed_csv/testresults_hazelnut_30000_9_BPT.csv
```

or

- CSV from your own computation:
```
results/hazelnut/testresults_hazelnut_30000_9_BPT.csv
```

Saves plots in:
```
results/boxplots/
```

---

### **5. `N3_Create_HTMLs.ipynb`**  
Creates full visual reports:

- Combined heatmap HTML  
- Combined IoU HTML  

Outputs:
```
results/imgs_hazelnut_heatmaps.html
results/imgs_hazelnut_iou.html
```

These pages allow quick inspection of explanations across the entire test set.

---

# 🖼️ Example Output

Below is an example for the **crack** anomaly category:

<table>
    <tr>
        <th>Input</th>
        <th>Recons</th>
        <th>Anomaly Map</th>
        <th>BPT-100</th>
        <th>BPT-500</th>
        <th>BPT-1000</th>
        <th>AA-100</th>
        <th>AA-500</th>
        <th>AA-1000</th>
        <th>LIME-250</th>
        <th>LIME-500</th>
        <th>LIME-1000</th>
        <th>GroundTruth</th>
    </tr>
    <tr>
        <td colspan="14" style="text-align: center;">
            <img src="results/test_results/crack/0_heatmaps_crack_0_30000_blend_2.png" width="100%">
            <img src="results/test_results/crack/0_IoU_crack_0_30000_blend_2.png" width="100%">
        </td>
    </tr>
</table>

The example shows:
- Input image  
- VAE-GAN reconstruction  
- Anomaly heatmap  
- ShapBPT (BPT-K) and AA-K explanations  
- LIME explanations  
- Ground-truth anomaly mask  
- IoU comparison  
