# Shapley image explanations with data-aware Binary Partition Trees

## Experiments on Facial Attributes Localization - E5
New Experiments E5, presented in supplementry materials

## Dataset:    
 - Dataset `CelebAMask-HQ` needs to be downloaded from [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).

# Model Setting:

1. Model with trained weights needed to download is provided by `kartikbatra` and publicaly [available here](https://www.kaggle.com/code/kartikbatra/multilabelclassification/notebook).
2. Make sure to download it and place it at [/models/model.pth](/models/model.pth)

# Folder Structure

1. Main folder `CelebA_experiments` contains notebooks and results folder.

2. Results folder contains further following folders:
    - Folders:
        - csv: folder contains [csv files](results/csv/) which are computed by Notebook: [`notebooks/CelebA_experiments/N1_Run_experiments_CelebA.ipynb`](`notebooks/CelebA_experiments/2_Run_experiments_CelebA.ipynb`) using different background replacement setup.
        - boxplots :   this folder contains pre-computed results presented in paper as `Figure 14`.
        - selected: This folder will contain the partial results used to generate the `Figure 13` of the paper.
    - Files: HTML-Files reporting visual analysis of computed explanations by various XAI methods.

## Notebooks 
Notebooks are available at `notebooks/CelebA_experiments/`

### Compute Explanations and CSV Files:
1. To run single selected examples, run [`notebooks/CelebA_experiments/N1_Run_experiments_CelebA.ipynb`](`notebooks/CelebA_experiments/2_Run_experiments_CelebA.ipynb`).
2. To run full test set for a specific background replacement values, select `background_type` first using `Cell#32`, and also by flipping the boolean variable `run_full_test` from `Cell#56` of [`notebooks/CelebA_experiments/N1_Run_experiments_CelebA.ipynb`](`notebooks/CelebA_experiments/2_Run_experiments_CelebA.ipynb`)
3. To run full ablation analysis on CelebA dataset using multiple background replacement values, flip boolean variable `run_ablation_analysis` in `Cell#59` & run `Cell#60` which will take almost 10 hours and will compute explanations and csv for all the background replacement values (i.e 'black' , 'white' , 'gray' , 'noise'  , 'full', 'blurred').

### Compute Results from Computed CSV and explanations:
1. To draw boxplots presented in supplementry material figures using [CSV files](results/csv/), run [`notebooks/CelebA_experiments/N2_DrawPlot_from_CSV.ipynb`](`notebooks/CelebA_experiments/N3_DrawPlot_from_CSV.ipynb`) by selecting the specific `background replacement value` in `Cell#7` and it will save the generated Boxplot at [Boxlpts Folder](results/boxplots/). (i.e. [Boxplot usiing blurred bg](results/boxplots/results_table_IoU_110_10_blurred.pdf))

3. All computed heatmap expalanations and IoU plots are combined togather as to generate HTML files to perform visual analysis run [`notebooks/CelebA_experiments/N3_Create_HTML_File.ipynb`](`notebooks/CelebA_experiments/N3_DrawPlot_from_CSV.ipynb`) which will generate [HTML Files for heatmaps](results/HTML_CelebA_blurred.html) & [HTML Files for IoU](results/HTML_CelebA_blurred_IoU.html) by loading the computed expalantions [`heatmaps~`](results/bg_blurred/) and by loading computed [CSV files](results/csv/csv_expIoU_face_116_10_blurred.csv).



## Example:
<div>
    <table>
    <tr>
    <td style="text-align: center;"> Input</td>
    <td style="text-align: center;"> GroundTruth</td>
    <td style="text-align: center;"> BPT-100</td>
    <td style="text-align: center;"> BPT-500</td>
    <td style="text-align: center;"> BPT-1000</td>
    <td style="text-align: center;"> AA-100</td>
    <td style="text-align: center;"> AA-500</td>
    <td style="text-align: center;"> AA-1000</td>
    <td style="text-align: center;"> LIME-50</td>
    <td style="text-align: center;"> LIME-100</td>
    <td style="text-align: center;"> LIME-200</td>
    <td style="text-align: center;"> GradShap</td>
<tr>
    <td colspan="12"><img src='results/bg_gray/0_Black_Hair_gray_heatmaps.png'></td>
<tr colspan="12">
<td colspan="12"><img src='results/bg_gray/0_Black_Hair_gray_iou.png'></td>
</table>
<div>
