# ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees

This repositry is provided as all tests (notebooks for various experiments, also listed below) for our paper **_ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees_** novel eXplainable AI (**_XAI_**) method **`ShapBPT`** accepted at **_`AAAI-2026`_** (40th Annual AAAI Conference on Artificial Intelligence) happening on 20-27 January 2026 in Singapore.

<html>


# Experiments Summary:
<!-- |  Name       | Dataset      | Model    | Short description      | Computation Time |
|  :--:       | :--:         |   :--:   | :--:                   | :--: |
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  |  7 hours 50 minutes |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | 4 hours 9 minutes | 
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     |  20 hours 6 minutes |
| E4          | MS-COCO      | Yolo11s  | Object detection     | 11 hours 42 minutes  |
| E5          | CelebA      | CNN  | Facial attributes localization     | 6 hours 14 minutes |  
| E6          | MVTec       | VAE-GAN  | Anomaly Detection     |  2 hours 56 minutes |
| E7          | ImageNet-S50      | ViT-Base  | Vision Transformer model     |  14 hours 48 minutes | -->


<!-- <p>This repo needs to be downloaded and merged at the same path which will place the models used to replicate the results.</p> -->

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Dataset</th>
      <th>Model</th>
      <th>Short description</th>
      <th>Model Path</th>
      <th>Computation Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>E1</td>
      <td>ImageNet-S<sub>50</sub></td>
      <td>ResNet50</td>
      <td>Common ImageNet setup</td>
      <td>Pretrained</td>
      <td> 7 hours 50 minutes </td>
    </tr>
    <tr>
      <td>E2</td>
      <td>ImageNet-S<sub>50</sub></td>
      <td>Ideal</td>
      <td>Controlled setup for exact IoU</td>
      <td>Pretrained</td>
      <td> 4 hours 9 minutes </td>
    </tr>
    <tr>
      <td>E3</td>
      <td>ImageNet-S<sub>50</sub></td>
      <td>SwinViT</td>
      <td>Vision Transformer model</td>
      <td>Pretrained</td>
      <td> 20 hours 6 minutes </td>
    </tr>
    <tr>
      <td>E4</td>
      <td>MS-COCO</td>
      <td>Yolo11s</td>
      <td>Object detection</td>
      <td><a href="/notebooks/E4_MS_COCO/yolo11s.pt"><strong><em>yolo11s.pt</em></strong></a></td>
      <td> 11 hours 42 minutes </td>
    </tr>
    <tr>
      <td>E5</td>
      <td>CelebA</td>
      <td>CNN</td>
      <td>Facial attributes localization</td>
      <td><a href="/notebooks/E5_CelebA/models/model.pth"><strong><em>model.pth</em></strong></a></td>
      <td> 6 hours 14 minutes </td>
    </tr>
    <tr>
      <td>E6</td>
      <td>MVTec</td>
      <td>VAE-GAN</td>
      <td>Anomaly Detection</td>
      <td><a href="/notebooks/E6_XAD/models/hazelnut_VAE_GAN_30000/"><strong><em>Pre trained VAE-GAN</em></strong></a></td>
      <td> 2 hours 56 minutes </td>
    </tr>
    <tr>
      <td>E7</td>
      <td>ImageNet-S<sub>50</sub></td>
      <td>ViT-Base16</td>
      <td>Vision Transformer model</td>
      <td>Pretrained</td>
      <td> 14 hours 48 minutes</td>
    </tr>
  </tbody>
</table>
</html>

<!-- # Precomputed Results:

  Precomputed CSV files and PDF files containing images for heatmaps are provided on this `ANONYMOUS LINK FOR RESULTS` $\rightarrow$ [**_`https://anonymous.4open.science/r/shapbpt_results`_**](https://anonymous.4open.science/r/shapbpt_experiments/) -->

<h2 style='color:red'> 1. Setup </h2>

<h3 style='color:white'> 1.1 create python environment </h3>


```bash
conda create -n env_shapbpt python==3.9.18
conda activate env_shapbpt
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Some code that generates the plots uses LaTeX to render text blocks. 
In order to run these code blocks, make sure to have `LaTeX` installed.
- for Ubuntu/Linux:
  ```
  sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
  ```
- for Windows:
  Install the MikTeX distribution (or equivalent).


<!-- ## Other XAI Methods -->
<h3 style='color:white'> 1.2 Other XAI Methods </h3>

- To reproduce the experimental results, a few additional XAI methods are required.
We used the `IDG` method from  [`saliencyMethods`](https://arxiv.org/pdf/2305.20052.pdf), which can be found at [`this link`](https://github.com/chasewalker26/Integrated-Decision-Gradients/tree/main/util/attribution_methods) (under BSD 3-Clause License). However, the IDG code is already included in `notebooks/utils/` folder, and no additional action is needed.


<h3 style='color:white'> 1.3 Download Tests Repo </h3>

```bash
git clone https://github.com/rashidrao-pk/shap_bpt_tests
cd shap_bpt_tests 
```

<h3 style='color:white'> 1.4 Download and install ShapBPT </h3>

```bash
git clone https://github.com/amparore/shap_bpt
```
- Follow the instructions mentioned in the repo [*_ShapBPT_*](https://github.com/amparore/shap_bpt) and make sure that the package is working.

```python
import shap_bpt as shap_bpt
print('shap_bpt version:',shap_bpt.__version__)
```



<!-- # 2. Retrieving the Dataset  -->

<h3 style='color:white'> 1.5. Datasets </h3>

Dataset needed to be downloaded are as belows;

|  Name       | Dataset Required       | Dataset Task Type | Download Link | Path | 
|  :--:       | :--:          |   :--:                    |      :--:      | :--: |
| E1          | ImageNet-S50  | Object Classification     | [**_ImageNet_**](https://www.image-net.org/) - [**_ImageNet-S_**](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset) | xx/xx |
| E2          | ImageNet-S50  | Object Classification     | [**_ImageNet_**](https://www.image-net.org/) - [**_ImageNet-S_**](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset) | xx/xx |
| E3          | ImageNet-S50  | Object Classification     | [**_ImageNet_**](https://www.image-net.org/) - [**_ImageNet-S_50_**](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset) |xx/xx |
| E4          | MS-COCO       | Multi Object Detection    | [**_MS-COCO SITE_**](https://cocodataset.org/#download) | xx/xx |
| E5          | CelebA        | Facial Features           | [**_CelebAMask-HQ_**](https://github.com/switchablenorms/CelebAMask-HQ) |  xx/xx |
| E6          | MVTec         | Visual Anomaly Detection  | [**_MVtec Link_**](https://www.mvtec.com/company/research/datasets/mvtec-ad) | xx/xx |
| E7          | ImageNet-S50  | Object Classification     | [**_ImageNet_**](https://www.image-net.org/) - [**_ImageNet-S_**](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset) | xx/xx |


Datasets including `ImageNet`, ImageNet-S<sub>50</sub>, `CelebA`, `MS-COCO`, and `MVTec` datasets are needed to reproduce the results presented in this paper. These datasets are publicly available, and are needed for running the complete experiments. 
Since we have no rights to redistribute them, you have to download them before running the experiments. 
So make sure to download these datsets in `notebook/datasets/` folder and they are  available as follows:
- **`ImageNet`** dataset (Validation-set) is required which can be downloaded from [**_ImageNet website_**](https://www.image-net.org/).
- **`ImageNetS-50`** is used for the experiments which requires to be downloaded from [**_this link_**](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset).
- **`Microsoft_COCO validation set (2017)`** can be downloaded from [**_this link_**](https://cocodataset.org/#download) together with the annotations (`instances_val2017.json`).
- **_`CelebA-Mask-HQ`_**: `CelebAMask-HQ` is a variant of [**Celeb-A dataset**](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and needs to be downloaded from [**_this link_**](https://github.com/switchablenorms/CelebAMask-HQ).  
- `MVTec Dataset`: This dataset is also publicaly available at  [**_this Link_**](https://www.mvtec.com/company/research/datasets/mvtec-ad).

------------------------------------------------------------------
<!-- # 3. Checking Precomputed Results -->


<h2 style='color:red'> 2. How to Use ShapBPT? </h2>

```python
# DEFINE EXPLAINER
explainer = shap_bpt.Explainer(f_masked, image_to_explain,num_explained_classes=4, verbose=True)
# EXPLAIN INSTANCE
shap_values_bpt = explainer.explain_instance(MAX_EVALS_BUDGET, # budget for explanation
                                              method='BPT', # Partitioning method (AA, or BPT)
                                              batch_size=batch_size,
                                              max_weight=max_weight)
#---------------------------------------------------------------------------
shap_values_aa = explainer.explain_instance(MAX_EVALS_BUDGET,
                                            method='AA',
                                            verbose_plot=False,
                                            batch_size=batch_size,
                                            max_weight=max_weight)
#---------------------------------------------------------------------------
shap_bpt.plot_owen_values(explainer, [shap_values_aa,shap_values_bpt], 
                          class_names, names=['AxisAligned','BPT'])

```


<h2 style='color:red'> 3. Checking Precomputed Results </h2>

## Precomputed Results for Experiments:
PDF FILES are also provided containing the precomputed results with images for each experiment.

|  Name       | Dataset      | Model    | Short description      | PDF file | CSV File |
|  :--:       | :--:         |   :--:   | :--:                   | :--: | :--: | 
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  |  [**_PDF-E1_**](PDF/HTML_E1_real_resnet_gray_combined.pdf) | [**_CSV-E1_**](results_logits_with_auc_clipped/csv_exp_E1_ImageNet_resnet_real_gray_logits.csv) |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | [**_PDF-E2_**](PDF/HTML_E2_ideal_resnet_gray_combined.pdf) | [**_CSV-E2_**](results_logits_with_auc_clipped/csv_exp_E2_ImageNet_resnet_ideal_gray_logits.csv) |
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     |  [**_PDF-E3_**](PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf) | [**_CSV-E3_**](results_logits_with_auc_clipped/csv_exp_E3_ImageNet_swin_trans_vit_real_gray_logits.csv) |
| E4          | MS-COCO      | Yolo11s  | Object detection     |  [**_PDF-E4_**](PDF/HTML_E4_yolo11s_gray_Combined.pdf)  | [**_CSV-E4_**](results_logits_with_auc_clipped/csv_exp_E4_yolo11s_gray_9.csv)  |
| E5          | CelebA      | CNN  | Facial attributes localization     | [**_PDF-E5_**](PDF/HTML_E5_CelebA_gray_combined.pdf) |   [**_CSV-E5_**](results_logits_with_auc_clipped/csv_exp_E5_IoU_face_1000_14_gray_brownhairs.csv) |  
| E6          | MVTec      | VAE-GAN  | Anomaly Detection     |  [**_PDF-E6_**](PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf) | [**_CSV-E6_**](results_logits_with_auc_clipped/csv_exp_E6_testresults_hazelnut_9_BPT_new_eval.csv) |
| E7          | ImageNet-S50      | ViT-base  | Vision Transformer     |  [**_PDF-E7_**](PDF/HTML_E7_ViT__combined_100.pdf) |  [**_CSV-E7_**](results_logits_with_auc_clipped/csv_exp_E7_ImageNet_vit_real_gray_logits.csv) |

------------------------------------------------------------------


<!-- # 4. Replication of the results and of the figures  -->
<h2 style='color:red'> 4. Replication of the results and of the figures </h2>

There are two ways to replicate the results preseneted in the paper:
  - `quick partial replication` that generates and replicate `Figure 1`, `Figure 3`, and `Figure 6`, 
  - `full replication` of the results being used to generate the results in CSV form for the experiments `E1`,`E2` and `E3`.

## a. Quick Partial Replication (partial results, takes a few minutes) 
Faster replication requires a few minutes to reproduce the claimed results:
 - Subfigures for `Figure 1` and `Figure 3` can be reproduced by running [`notebooks/N1_Fig1_and_Fig3.ipynb`](notebooks/N1_Fig1_and_Fig3.ipynb) and the generated figures will be saved at [`/notebooks/paper_figures`](/notebooks/paper_figures) which are used to combined to generate `Figure 1` and `Figure 3`.
- `Figure 4 (E1 & E2)` can be reproduced by running [`notebooks/N2_DrawPlotFig4_from_CSV.ipynb`](notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb) which will simply load the already computed CSV files;
  - `Figure 4 (E1)`: using [`CSV file from resnet model`](/notebooks/E1_E2_E3_E7/CSV/csv_exp_ImageNetS_real_gray.csv) 
  - `Figure 4 (E2)`: using [`CSV file from linear model`](/notebooks/E1_E2_E3_E7/CSV/csv_exp_ImageNetS_ideal_gray.csv)
- `Figure 5` in supplementery material are a selection of images generated by using `ALL/SELECTED IMAGES TEST'` cell in [`notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb`](notebooks/E1_E2_E3_E7/N1_Run_experiments/.ipynb) at `notebooks/results/ImageNetS/{model}_{background}/selected`.
- `Figure 6` can be reproduced by running [`notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb`](notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb) which will simply load the already computed CSV file:
  - `Figure 6 (E3)`: using [`CSV file for mutiple_backgrounds`](/notebooks/E1_E2_E3_E7/CSV/csv_exp_ImageNetS_real_full.csv) 


## b. Full Replication of the paper results (takes about 2 days)
Replication for the full test set can take:
- **Experiment E1:** `model=resnet` & `background=gray` takes about **24 hours**
- **Experiment E2:** `model=ideal`                      takes about **16 hours**
- **Experiment E3:** `model=resnet` & `background=full` takes about **30 hours**

To Replicate the results, following codes are required;
 - `Figure 4`, is a selection of examples which are generated using [`notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb`](notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb).
 - To run the test on all images for ImageNetS-50, run [`notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb`](notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb). This notebook will save saliency-map images and IoU images and CSV file as per selected two main parameters 
    - `model_type='real'` computed the results/csv with `ResNet-50` model. 
    - `model_type='ideal'` computes results/csv with `linear/ideal` model.
    - `background_type='gray'` computes results/csv with background replacement values with a `solid gray background`.
 - After running [`notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb`](notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb), generated heatmaps images can be converted to `Webpage` having visual and numerical results using [`notebooks/E1_E2_E3_E7/N3_Create_HTML_File.ipynb`](notebooks/E1_E2_E3_E7/N3_Create_HTML_File.ipynb) within `notebooks/E1_E2_E3_E7/additional_material` folder.
- **Experiment E4:** `model=resnet` & `background=full` takes about **30 hours**
- **Experiment E5:** `model=resnet` & `background=full` takes about **30 hours**

<!-- # 4. Precomputed CSV & Heatmaps
  All results/boxplots presented in technical appendix is provided at this `ANONYMOUS LINK FOR RESULTS` $\rightarrow$ [**_`https://anonymous.4open.science/r/shapbpt_results`_**](https://anonymous.4open.science/r/shapbpt_experiments/) 
  This link contains Precomputed
  - CSV to generate plots again
  - PDF files containing images for heatmaps. -->


**Note**: The published results were generated and validated on two hardwares with different architecture:
<hr>

| Device Type | Machine | Processor       | RAM  | GPU | 
| :--:        | :--    | :--            |   :--: |            :--:           |
| LAPTOP      | Santech XN2  | Corei9 13th Gen | 16GB | NVIDIA GeForce RTX 4070 |
| LAPTOP      | Apple Macbook Pro  | Apple M1        | 16GB   | M1 GPU   




### structure

 ```
├── ShapBPT_Experiments
|   ├── notebooks
|   |   ├── E1_E2_E3_E7
|   |   |   ├── ---
|   |   ├── E4_MS_COCO
|   |   |   ├── ---
|   |   ├── E5_CelebA
|   |   |   ├── ---
|   |   ├── E6_XAD
|   |   |   ├── ---
|   ├── PDF
|   |   |   ├── README.md
|   |   |   ├── HTML_E1_real_resnet_gray_combined.pdf
|   |   |   ├── HTML_E2_ideal_resnet_gray_combined.pdf
|   |   |   ├── HTML_E3_real_swin_trans_vit_gray_combined.pdf
|   |   |   ├── HTML_E4_yolo11s_gray_Combined.pdf
|   |   |   ├── HTML_E5_CelebA_gray_combined.pdf
|   |   |   ├── HTML_E6_hazelnut_heatmaps_IoU.pdf
|   ├── results_logits_with_auc_clipped
|   |   |   ├── README.md
|   |   |   ├── csv_exp_E1_ImageNet_resnet_real_gray_logits.csv
|   |   |   ├── csv_exp_E2_ImageNet_resnet_ideal_gray_logits.csv
|   |   |   ├── csv_exp_E3_ImageNet_swin_trans_vit_real_gray_logits.csv
|   |   |   ├── csv_exp_E4_yolo11s_gray_9.csv
|   |   |   ├── csv_exp_E5_IoU_face_1000_14_gray_brownhairs.csv
|   |   |   ├── csv_exp_E6_testresults_hazelnut_9_BPT_new_eval.csv
|   |   |   ├── csv_exp_E7_ImageNet_vit_real_gray_logits.csv
|   ├── saved_explanations
|   |   |   ├── ----
|   |   |   ├── ----

 ```


# Contributing
If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at contact@website.com or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.

<!-- cite
```md
ShapBPT: Image Feature attributions using Data Aware Binary Partition Trees. 40th AAAI conference on Artificial Intelligence.
``` -->