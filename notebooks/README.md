# ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees

This reposirty contains the experiments presented in the **_ShapBPT: Image Feature Attributions using Data-Aware Binary Partition Trees_**, a novel eXplainable AI (**_XAI_**) method [**`ShapBPT`**](ShapBPT/README.md) accpted in AAAI-26.


<!-- It contains the material needed to install the package, run and replicate the experiments, and generate the figures/plots included in the paper. -->

<!-- <img src="ShapBPT/docs/explain_bpt.svg"> -->

<!-- # What is included -->


<!-- #### Summary of experiments -->

<!-- # Experiments Details:
|  Name       | Dataset      | Model    | Short description      | Computation Time |
|  :--:       | :--:         |   :--:   | :--:                   | :--: |
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  |  7 hours 50 minutes |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | 4 hours 9 minutes | 
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     |  20 hours 6 minutes |
| E4          | MS-COCO      | Yolo11s  | Object detection     | 11 hours 42 minutes  |
| E5          | CelebA      | CNN  | Facial attributes localization     | 6 hours 14 minutes |  
| E6          | MVTec       | VAE-GAN  | Anomaly Detection     |  2 hours 56 minutes |
| E7          | ImageNet-S50      | ViT-Base  | Vision Transformer model     |  14 hours 48 minutes | -->


This folder contains all notebook folders for various applications of **_ShapBPT_** including:

**Folders**:
1. [***`E1_to_E3`***](E1_to_E3) contains ShapBPT examples and codes for `Image localization problem` on [`ImageNet`](#) dataset.
2. [***`E4_MS_COCO`***](E4_MS_COCO) contains ShapBPT examples and codes for `Object detection Task` on [`Microsoft-COCO`](https://cocodataset.org) dataset.
3. [***`E5_CelebA`***](E5_CelebA) contains ShapBPT examples and codes for `Facial Attribute Localization` on [`CelebA Dataset`](#) dataset.
4. [***`E6_XAD`***](E6_XAD) contains ShapBPT examples and codes for `Visual Anomaly Deetction task` on [`MVTec dataset`](#) dataset.

5. [`dataset`](dataset/) should contain all the required datasets to place in it.
6. [`paper_figures`](paper_figures/) contains all subfigures required to generate `Figure_1` & `Figure_2` of the paper.
7. [`utils`](utils/) contains supporting files needs for the notebooks.



<!-- ## Instrctions
```bash
git clone https://github.com/amparore/shap_bpt
```
- Follow the instructions mentioned in the repo and make sure that the package is working.

```python
import shapbpt
print(shapbpt.__version__)
```


## DOWNALOD AND INTEGRATE PRECOMPUTED RESULTS
```bash
mkdir shapbpt
cd shapbpt
git clone https://github.com/r4sshd/shapbpt_results
mv shapbpt_results 
``` -->

<!-- - Additional description of formulas, theorems and experiments [**`Supplementary.pdf`**](Supplementary.pdf) -->
<!-- - Environment preparation and Installation instructions. -->
<!-- - Instructions to retrieve the datasets. -->
<!-- - Replication instructions (Quick Partial Replication, Full Replication). -->
<!-- - Precomputed results for all experiments ([**`PDF files`**](/notebooks/PDF) ). -->
<!-- <br> -->

<!-- # 1.  Environment preparation and Installation instructions
The first thing is to create `python environment` where `shap_bpt` can be installed properly and notebooks can be run.

## a. Required Python packages

To create a new python environment to experiment with `ShapBPT`, run the commands:

```cmd
conda create -n env_shapbpt python==3.9.18
conda activate env_shapbpt
pip install -r requirements.txt
``` -->


<!-- Moreover, to test it on GPU, following command needs running also:
```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -->
<!-- ``` -->
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

  Alternativly, Follow the instruction on [`this page`](https://github.com/nuncjo/cython-installation-windows)

## c. Build and install ShapBPT
A [**_`Cython`_**](https://cython.org/) working environment is needed to build the package.
ShapBPT contains a `cython` module, that needs to be compiled separately, before installing the `shap_bpt` python module.

### Build instructions

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
  python setup.py clean --all  # To clean up the folder from the intermediate build files
  ```


Some code that generates the plots uses LaTeX to render text blocks. 
In order to run these code blocks, make sure to have `LaTeX` installed.
- for Ubuntu/Linux:
  ```
  sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
  ```
- for Windows:
  Install the MikTeX distribution (or equivalent). -->


<!-- ## Run the Notebooks to replicate the paper's results
To start Jupyter and run the notebooks, type in the shell
```cmd
jupyter notebook
``` -->


<!-- ## c. Configure other XAI Methods
- To reproduce the experimental results, a few additional XAI methods are required.
We used the `IDG` method from  [`saliencyMethods`](https://arxiv.org/pdf/2305.20052.pdf), which can be found at [`this link`](https://github.com/chasewalker26/Integrated-Decision-Gradients/tree/main/util/attribution_methods) (under BSD 3-Clause License). However, the IDG code is already included in `notebooks/utils/` folder, and no additional action is needed.

# 2. Retrieving the Dataset 
Datasets including `ImageNet`, ImageNet-S<sub>50</sub>, `CelebA`, `MS-COCO`, and `MVTec` datasets are needed to reproduce the results presented in this paper. These datasets are publicly available, and are needed for running the complete experiments. 
Since we have no rights to redistribute them, you have to download them before running the experiments. 
So make sure to download these datsets in `notebook/datasets/` folder and they are  available as follows:
- **`ImageNet`** dataset (Validation-set) is required which can be downloaded from [**_ImageNet website_**](https://www.image-net.org/).
- **`ImageNetS-50`** is used for the experiments which requires to be downloaded from [**_this link_**](https://github.com/LUSSeg/ImageNet-S?tab=readme-ov-file#get-part-of-the-imagenet-s-dataset).
- **`Microsoft_COCO validation set (2017)`** can be downloaded from [**_this link_**](https://cocodataset.org/#download) together with the annotations (`instances_val2017.json`).
- **_`CelebA-Mask-HQ`_**: `CelebAMask-HQ` is a variant of [**Celeb-A dataset**](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and needs to be downloaded from [**_this link_**](https://github.com/switchablenorms/CelebAMask-HQ).  
- `MVTec Dataset`: This dataset is also publicaly available at  [**_this Link_**](https://www.mvtec.com/company/research/datasets/mvtec-ad).

# 3. Retreiving the Models:

- To replicate the results, there are 3 Pretrained models including yolov11, CelebA-Model, and AD_Hazelnut model needs to be downloaded from this `ANONYMOUS LINK FOR MODELS` $\rightarrow$ [**_`https://anonymous.4open.science/r/shapbpt_experiments`_**](https://anonymous.4open.science/r/shapbpt_experiments/), download the whole repo and merge with existing one which will place models only into this relevant folders. -->

<!-- # 3. Replication of the results and of the figures  -->
<!-- There are two ways to replicate the results preseneted in the paper:
  - `quick partial replication` that generates and replicate `Figure 1`, `Figure 3`, and `Figure 6`, 
  - `full replication` of the results being used to generate the results in CSV form for the experiments `E1`,`E2` and `E3`. -->

<!-- ## a. Quick Partial Replication (partial results, takes a few minutes) 
Faster replication requires a few minutes to reproduce the claimed results:
 - Subfigures for `Figure 1` and `Figure 3` can be reproduced by running [`notebooks/N1_Fig1_and_Fig3.ipynb`](notebooks/N1_Fig1_and_Fig3.ipynb) and the generated figures will be saved at [`/notebooks/paper_figures`](/notebooks/paper_figures) which are used to combined to generate `Figure 1` and `Figure 3`.
- `Figure 4 (E1 & E2)` can be reproduced by running [`notebooks/N2_DrawPlotFig4_from_CSV.ipynb`](notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb) which will simply load the already computed CSV files;
  - `Figure 4 (E1)`: using [`CSV file from resnet model`](/notebooks/E1_E2_E3_E7/CSV/csv_exp_ImageNetS_real_gray.csv) 
  - `Figure 4 (E2)`: using [`CSV file from linear model`](/notebooks/E1_E2_E3_E7/CSV/csv_exp_ImageNetS_ideal_gray.csv)
- `Figure 5` in supplementery material are a selection of images generated by using `ALL/SELECTED IMAGES TEST'` cell in [`notebooks/E1_E2_E3_E7/N1_Run_experiments.ipynb`](notebooks/E1_E2_E3_E7/N1_Run_experiments/.ipynb) at `notebooks/results/ImageNetS/{model}_{background}/selected`.
- `Figure 6` can be reproduced by running [`notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb`](notebooks/E1_E2_E3_E7/N2_DrawPlotFig4_Fig6_from_CSV.ipynb) which will simply load the already computed CSV file:
  - `Figure 6 (E3)`: using [`CSV file for mutiple_backgrounds`](/notebooks/E1_E2_E3_E7/CSV/csv_exp_ImageNetS_real_full.csv)  -->

<!-- 
## b. Full Experiments of the results (~ 2 days)
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

# 4. Precomputed CSV & Heatmaps
  All results/boxplots presented in technical appendix is provided at this `ANONYMOUS LINK FOR RESULTS` $\rightarrow$ [**_`https://anonymous.4open.science/r/shapbpt_results`_**](https://anonymous.4open.science/r/shapbpt_experiments/) 
  This link contains Precomputed
  - CSV to generate plots again
  - PDF files containing images for heatmaps. -->
   
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

<br/>

<!-- # Experiments Details:
|  Name       | Dataset      | Model    | Short description      | Computation Time |
|  :--:       | :--:         |   :--:   | :--:                   | :--: |
| E1          | ImageNet-S50 | ResNet50 | Common ImageNet setup  |  7 hours 50 minutes |
| E2          | ImageNet-S50 | Ideal    | Controlled setup for exact IoU | 4 hours 9 minutes | 
| E3          | ImageNet-S50 | SwinViT  | Vision Transformer model     |  20 hours 6 minutes |
| E4          | MS-COCO      | Yolo11s  | Object detection     | 11 hours 42 minutes  |
| E5          | CelebA      | CNN  | Facial attributes localization     | 6 hours 14 minutes |  
| E6          | MVTec       | VAE-GAN  | Anomaly Detection     |  2 hours 56 minutes |
| E7          | ImageNet-S50      | ViT-Base  | Vision Transformer model     |  14 hours 48 minutes | -->
<!-- | E8          | MVTec      | VAE-GAN  | Anomaly Detection     |  [**_PDF-E6_**](/notebooks/PDF/HTML_E6_hazelnut_heatmaps_IoU.pdf) | -->

<!-- | E3          | ImageNet-S50 | ResNet50 | Multiple replacement values     |   -->

<!-- 
**Note**: The published results were generated and validated on two hardwares with different architecture:
<hr>

<!-- | Device Type | Machine | Processor       | RAM  | GPU | 
| :--:        | :--    | :--            |   :--: |            :--:           |
| LAPTOP      | Santech XN2  | Corei9 13th Gen | 16GB | NVIDIA GeForce RTX 4070 |
| LAPTOP      | Apple Macbook Pro  | Apple M1        | 16GB   | M1 GPU                  | -->


<!-- # Contributing
If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at contact@website.com or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license. -->
