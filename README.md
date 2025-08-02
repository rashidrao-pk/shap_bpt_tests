# ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees

This package provided as Supplementary Material for the **_ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees_** implementation of novel eXplainable AI (**_XAI_**) method **`ShapBPT`**.

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

  Precomputed CSV files and PDF files containing images for heatmaps are provided on this `ANONYMOUS LINK FOR RESULTS` $\rightarrow$ [**_`https://anonymous.4open.science/r/shapbpt_results`_**](https://anonymous.4open.science/r/shapbpt_experiments/)





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