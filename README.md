# Shapley image explanations with data-aware Binary Partition Trees

This package provided as Supplementary Material for the **_Shapley image explanations with data-aware Binary Partition Trees_** implementation of novel eXplainable AI (**_XAI_**) method **`ShapBPT`**.

<html>
<h2 style='color:red'> Pre-trained Models </h2>


<p>This repo needs to be downloaded and merged at the same path which will place the models used to replicate the results.</p>

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Dataset</th>
      <th>Model</th>
      <th>Short description</th>
      <th>Model Path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>E1</td>
      <td>ImageNet-S50</td>
      <td>ResNet50</td>
      <td>Common ImageNet setup</td>
      <td>Pretrained</td>
    </tr>
    <tr>
      <td>E2</td>
      <td>ImageNet-S50</td>
      <td>Ideal</td>
      <td>Controlled setup for exact IoU</td>
      <td>Pretrained</td>
    </tr>
    <tr>
      <td>E3</td>
      <td>ImageNet-S50</td>
      <td>SwinViT</td>
      <td>Vision Transformer model</td>
      <td>Pretrained</td>
    </tr>
    <tr>
      <td>E4</td>
      <td>MS-COCO</td>
      <td>Yolo11s</td>
      <td>Object detection</td>
      <td><a href="/notebooks/E4_MS_COCO/yolo11s.pt"><strong><em>yolo11s.pt</em></strong></a></td>
    </tr>
    <tr>
      <td>E5</td>
      <td>CelebA</td>
      <td>CNN</td>
      <td>Facial attributes localization</td>
      <td><a href="/notebooks/E5_CelebA/models/model.pth"><strong><em>model.pth</em></strong></a></td>
    </tr>
    <tr>
      <td>E6</td>
      <td>MVTec</td>
      <td>VAE-GAN</td>
      <td>Anomaly Detection</td>
      <td><a href="/notebooks/E6_XAD/models/hazelnut_VAE_GAN_30000/"><strong><em>Pre trained VAE-GAN</em></strong></a></td>
    </tr>
    <tr>
      <td>E7</td>
      <td>ImageNet-S<sub>50</sub></td>
      <td>ViT-Base16</td>
      <td>Vision Transformer</td>
      <td>Pretrained</td>
    </tr>
  </tbody>
</table>
</html>

# Precomputed Results:


## Precomputed CSV files:

Results after running all the experiments are provided in CSV files given below.

|  Name       | Dataset      | Model       | Pre-computed CSV file |
|  :--:       | :--:         |   :--:                     | :--: |
| E1          | ImageNet-S50 | ResNet50   |  [**_CSV-E1_**](/results_logits_with_auc_clipped/csv_exp_E1_ImageNet_resnet_real_gray_logits.csv) |
| E2          | ImageNet-S50 | Ideal    | [**_CSV-E2_**](/results_logits_with_auc_clipped/csv_exp_E2_ImageNet_resnet_ideal_gray_logits.csv) | 
| E3          | ImageNet-S50 | SwinViT  |  [**_CSV-E3_**](/results_logits_with_auc_clipped/csv_exp_E3_ImageNet_swin_trans_vit_real_gray_logits.csv) |
| E4          | MS-COCO      | Yolo11s  | [**_CSV-E4_**](/results_logits_with_auc_clipped/csv_exp_E4_yolo11s_gray_9.csv)  |
| E5          | CelebA      | CNN  | [**_CSV-E5_**](/results_logits_with_auc_clipped/csv_exp_E5_IoU_face_1000_14_gray_brownhairs.csv) |  
| E6          | MVTec      | VAE-GAN  |  [**_CSV-E6_**](/results_logits_with_auc_clipped/csv_exp_E6_testresults_hazelnut_9_BPT_new_eval.csv) |
| E7          | ImageNet-S50      | ViT-base  |  [**_CSV-E7_**](/results_logits_with_auc_clipped/csv_exp_E7_ImageNet_vit_real_gray_logits.csv) |


**_`saved_explanations`_** is containing the saved explanations for Paper Figures and tese explanations were saved by running **_`N1_Fig1_and_Fig3.ipynb`_**

<hr>

## Precomputed Results on Experiments:
PDF FILES are also provided containing the precomputed results with images for each experiment.

|  Name       | Dataset      | Model    | Short description      | Pre-computed results file |
|  :--:       | :--:         |   :--:   | :--:                   | :--: |
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  |  [**_PDF-E1_**](/notebooks/PDF/HTML_E1_real_resnet_gray_combined.pdf) |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | [**_PDF-E2_**](/notebooks/PDF/HTML_E2_ideal_resnet_gray_combined.pdf) | 
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     |  [**_PDF-E3_**](/notebooks/PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf) |
| E4          | MS-COCO      | Yolo11s  | Object detection     | [**_PDF-E4_**](/notebooks/PDF/HTML_E4_yolo11s_gray_Combined.pdf)  |
| E5          | CelebA      | CNN  | Facial attributes localization     | [**_PDF-E5_**](/notebooks/PDF/HTML_E5_CelebA_gray_combined.pdf) |  
| E6          | MVTec      | VAE-GAN  | Anomaly Detection     |  [**_PDF-E6_**](/notebooks/PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf) |





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
|   |   |   |   ├── train-semi-segmentation
|   |   |   |   |   ├── n01443537
|   |   |   |   |   |   ├── ILSVRC2012_val_00004677.png
|   |   |   |   |   |   ├── ILSVRC2012_val_00048864.png
|   |   |   |   |   ├── n01491361
|   |   |   |   |   ├── n12998815
 ```