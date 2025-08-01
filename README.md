# Shapley image explanations with data-aware Binary Partition Trees

This package provided as Supplementary Material for the **_Shapley image explanations with data-aware Binary Partition Trees_** implementation of novel eXplainable AI (**_XAI_**) method **`ShapBPT`**.


## Pre-trained Models:
This repo needs be downloaded and merged at the same path which will place the models used to replicate the results.
|  Name       | Dataset      | Model    | Short description      | Model Path |
|  :--:       | :--:         |   :--:   | :--:                   | :--: |
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  | Pretrained  |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | Pretrained | 
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     | Pretrained  |
| E4          | MS-COCO      | Yolo11s  | Object detection     |  [**_`yolo11s.pt`_**](/notebooks/E4_MS_COCO/yolo11s.pt) |
| E5          | CelebA      | CNN  | Facial attributes localization     | [**_`model.pth`_**](/notebooks/E5_CelebA/models/model.pth) |  
| E6          | MVTec      | VAE-GAN  | Anomaly Detection     | [**_`Pre trained VAE-GAN`_**](/notebooks/E6_XAD/models/hazelnut_VAE_GAN_30000/*)  |





$$\alpha \rightarrow \sigma$$


<html>
<h2 style='color:red'> Hello </h2>

</html>
<!-- 
# What is included

- Additional description of formulas, theorems and experiments [**`Supplementary.pdf`**](Supplementary.pdf)
- Environment preparation and Installation instructions.
- Instructions to retrieve the datasets.
- Replication instructions (Quick Partial Replication, Full Replication).
- Precomputed results for all experiments ([**`PDF files`**](/notebooks/PDF) ).
<br>

# 1.  Environment preparation and Installation instructions
The first thing is to create `python environment` where `shap_bpt` can be installed properly and notebooks can be run.

## a. Required Python packages

To create a new python environment to experiment with `ShapBPT`, run the commands:

```cmd
conda create -n env_shapbpt python==3.9.18
conda activate env_shapbpt
pip install -r requirements.txt
```

Moreover, to test it on GPU, following command needs running also:
```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
<!--
conda create -n env_shapbpt
conda activate env_shapbpt
conda config --env --set subdir win-32   # Windows only
conda install python=3.9
pip install -r requirements.txt
-->
<!-- ## b. Preparing the Cython environment
A working C compiler (for Windows we tested the mingw32 environment) and Cython needs to be installed. 
  - On Debian-based `Linux` systems run
    ```
    sudo apt install build-essential
    ```
  - On `macOS`, install the `Apple Developer Tools`

- On Windows system:

  **Recommended:** To install mingw using conda commands recommended on [`this page`](https://python-at-risoe.pages.windenergy.dtu.dk/compiling-on-windows/configuration.html) to setup a working mingw32 system, Run following lines.
    ```cmd
    conda install numpy libpython m2w64-toolchain cython
    ```
  **Note**: *Make sure that environment is activated before running above line of code*.

  Alternativly, Follow the instruction on [`this page`](https://github.com/nuncjo/cython-installation-windows) -->
<!-- 
## c. Build and install ShapBPT
A [**_`Cython`_**](https://cython.org/) working environment is needed to build the package.
ShapBPT contains a `cython` module, that needs to be compiled separately, before installing the `shap_bpt` python module. -->

<!-- ### Build instructions

- Change current directory to ShapBPT folder
  ```cmd
  cd ShapBPT
  ```
- On `Unix` systems, run following command:
  ```cmd
  python setup.py build_ext --inplace
  ```

- On `Windows` systems, run following command:
  ```
  python setup.py build_ext --inplace --compiler=mingw32
  ```

- After compiling, the ShapBPT python module can be installed using:
  ```bash
  python setup.py install      # install ShapBPT
  python setup.py clean --all  # To clean up the folder from the intermediate build files -->
  <!-- ``` -->

<!-- 
## Run the Notebooks to replicate the paper's results
To start Jupyter and run the notebooks, type in the shell
```cmd
jupyter notebook
```
Some code that generates the plots uses LaTeX to render text blocks. 
In order to run these code blocks, make sure to have `LaTeX` installed.
- for Ubuntu/Linux:
  ```
  sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
  ```
- for Windows:
  Install the MikTeX distribution (or equivalent). -->

<!-- ## c. Other XAI Methods
- To reproduce the experimental results, a few additional XAI methods are required.
We used the `IDG` method from  [`saliencyMethods`](https://arxiv.org/pdf/2305.20052.pdf), which can be found at [`this link`](https://github.com/chasewalker26/Integrated-Decision-Gradients/tree/main/util/attribution_methods) (under BSD 3-Clause License). However, the IDG code is already included in `notebooks/utils/` folder, and no additional action is needed. -->

<!-- # 2. Retrieving the Dataset 
The ImageNet and ImageNet-S dataset are needed to reproduce the results presented in this paper. These datasets are publicly available, and are needed for running the complete experiments. 
Since we have no rights to redistribute them, you have to download them before running the experiments. 
So make sure to download these datsets in `notebook/dataset/imageNet` and `notebook/dataset/imageNetS` folders respectively and they are  available as follows:
- **`ImageNet`** dataset (Validation-set) is required which can be downloaded from [`ImageNet website`](https://www.image-net.org/)
- **`ImageNetS-50`** is used for the experiments which requires to be downloaded from [`this link`](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset)
- **`Microsoft_COCO validation set (2017)`** can be downloaded from [**_this link_**](https://cocodataset.org/#download) together with the annotations (`instances_val2017.json`). -->

<!-- # 3. Retreiving the Models:

- Models needs to be downloaded from this **[ANONYMOUS LINK FOR MODELS](#)**. -->


<!-- 
# 3. Replication of the results and of the figures 
There are two ways to replicate the results preseneted in the paper:
  - `quick partial replication` that generates and replicate `Figure 1`, `Figure 3`, and `Figure 6`, 
  - `full replication` of the results being used to generate the results in CSV form for the experiments `E1`,`E2` and `E3`. -->

<!-- ## a. Quick Partial Replication (partial results, takes a few minutes) 
Faster replication requires a few minutes to reproduce the claimed results:
 - Subfigures for `Figure 1` and `Figure 3` can be reproduced by running [`notebooks/N1_Fig1_and_Fig3.ipynb`](notebooks/N1_Fig1_and_Fig3.ipynb) and the generated figures will be saved at [`/notebooks/paper_figures`](/notebooks/paper_figures) which are used to combined to generate `Figure 1` and `Figure 3`.
- `Figure 4 (E1 & E2)` can be reproduced by running [`notebooks/N2_DrawPlotFig4_from_CSV.ipynb`](notebooks/E1_to_E3/N2_DrawPlotFig4_Fig6_from_CSV.ipynb) which will simply load the already computed CSV files;
  - `Figure 4 (E1)`: using [`CSV file from resnet model`](/notebooks/E1_to_E3/CSV/csv_exp_ImageNetS_real_gray.csv) 
  - `Figure 4 (E2)`: using [`CSV file from linear model`](/notebooks/E1_to_E3/CSV/csv_exp_ImageNetS_ideal_gray.csv)
- `Figure 5` in supplementery material are a selection of images generated by using `ALL/SELECTED IMAGES TEST'` cell in [`notebooks/E1_to_E3/N1_Run_experiments.ipynb`](notebooks/E1_to_E3/N1_Run_experiments/.ipynb) at `notebooks/results/ImageNetS/{model}_{background}/selected`.
- `Figure 6` can be reproduced by running [`notebooks/E1_to_E3/N2_DrawPlotFig4_Fig6_from_CSV.ipynb`](notebooks/E1_to_E3/N2_DrawPlotFig4_Fig6_from_CSV.ipynb) which will simply load the already computed CSV file:
  - `Figure 6 (E3)`: using [`CSV file for mutiple_backgrounds`](/notebooks/E1_to_E3/CSV/csv_exp_ImageNetS_real_full.csv)  -->


<!-- ## b. Full Replication of the paper results (takes about 2 days)
Replication for the full test set can take:
- **Experiment E1:** `model=resnet` & `background=gray` takes about **24 hours**
- **Experiment E2:** `model=ideal`                      takes about **16 hours**
- **Experiment E3:** `model=resnet` & `background=full` takes about **30 hours**

To Replicate the results, following codes are required;
 - `Figure 4`, is a selection of examples which are generated using [`notebooks/E1_to_E3/N1_Run_experiments.ipynb`](notebooks/E1_to_E3/N1_Run_experiments.ipynb).
 - To run the test on all images for ImageNetS-50, run [`notebooks/E1_to_E3/N1_Run_experiments.ipynb`](notebooks/E1_to_E3/N1_Run_experiments.ipynb). This notebook will save saliency-map images and IoU images and CSV file as per selected two main parameters 
    - `model_type='real'` computed the results/csv with `ResNet-50` model. 
    - `model_type='ideal'` computes results/csv with `linear/ideal` model.
    - `background_type='gray'` computes results/csv with background replacement values with a `solid gray background`.
 - After running [`notebooks/E1_to_E3/N1_Run_experiments.ipynb`](notebooks/E1_to_E3/N1_Run_experiments.ipynb), generated heatmaps images can be converted to `Webpage` having visual and numerical results using [`notebooks/E1_to_E3/N3_Create_HTML_File.ipynb`](notebooks/E1_to_E3/N3_Create_HTML_File.ipynb) within `notebooks/E1_to_E3/additional_material` folder. -->


  <!-- <details>
  <summary> Click here to check the paths for all precomputed results results: </summary> -->


  <!-- ## _Results on selected Set of Images_
  |  Name       | Dataset      | Model    | Short description    | _LINK_ |
  |  :--:          |   :--:   | :--:                   | :--:                   | :--:                   |  
  | E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  | [**_PDF-E1_**](/notebooks/PDF/HTML_E1_real_resnet_gray_combined.pdf) | 
  | E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU  | [**_PDF-E2_**](/notebooks/PDF/HTML_E2_ideal_resnet_gray_combined.pdf) |
  | E3          | ImageNet-S50 | SwinViT  | Vision Transformer model  | [**_PDF-E3_**](/notebooks/PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf) |
  | E4          | MS-COCO      | Yolo11s  | Facial attributes localization      | [**_PDF-E4_**](/notebooks/PDF/HTML_E4_yolo11s_gray_Combined.pdf) |
   | E5          | CelebA      | CNN  | Object detection      | [**_PDF-E5_**](/notebooks/PDF/HTML_E5_CelebA_gray_combined.pdf) |
    | E6          | MVTec      | VAE-GAN  | Explainable Anomaly Detection      | [**_PDF-E6_**](/notebooks/PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf) | -->

  <!-- </details> -->

<!-- <br/>

## Experiments Details:
|  Name       | Dataset      | Model    | Short description      | Pre-computed results file |
|  :--:       | :--:         |   :--:   | :--:                   | :--: |
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  |  [**_PDF-E1_**](/notebooks/PDF/HTML_E1_real_resnet_gray_combined.pdf) |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | [**_PDF-E2_**](/notebooks/PDF/HTML_E2_ideal_resnet_gray_combined.pdf) | 
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     |  [**_PDF-E3_**](/notebooks/PDF/HTML_E3_real_swin_trans_vit_gray_combined.pdf) |
| E4          | MS-COCO      | Yolo11s  | Object detection     | [**_PDF-E4_**](/notebooks/PDF/HTML_E4_yolo11s_gray_Combined.pdf)  |
| E5          | CelebA      | CNN  | Facial attributes localization     | [**_PDF-E5_**](/notebooks/PDF/HTML_E5_CelebA_gray_combined.pdf) |  
| E6          | MVTec      | VAE-GAN  | Anomaly Detection     |  [**_PDF-E6_**](/notebooks/PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf) | -->

<!-- | E3          | ImageNet-S50 | ResNet50 | Multiple replacement values     |   -->

<!-- <hr> -->

<!-- **Note**: The published results were generated and validated on two hardwares with different architecture:
| Device Type | Machine | Processor       | RAM  | GPU | 
| :--:        | :--    | :--            |   :--: |            :--:           |
| LAPTOP      | Santech XN2  | Corei9 13th Gen | 16GB | NVIDIA GeForce RTX 4070 |
| LAPTOP      | Apple Macbook Pro  | Apple M1        | 16GB   | M1 GPU                  | -->


<!-- # Contributing
If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at contact@website.com or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license. --> 
