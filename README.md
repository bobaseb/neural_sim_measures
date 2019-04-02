# Code for Measures of Neural Similarity (https://doi.org/10.1101/439893)

## System Requirements to fully analyze the data:

Python 2.7, PyMVPA 2.6.4, Scikit-learn 0.19.1, Scipy 1.0.0, FMRIB Software Library (FSL) 5.0.9, R 3.2.3, lme4 1.1-17, multcomp 1.4-8

## Hardware requirements:

Optimizing feature selection and running searchlights can be slow on an average computer. Using a cluster is recommended.

## Installation:

After installing all the system requirements above, just download the source code and change the directories in the code according to your local directories.

## Further Instructions:

1) Data on OSF (https://osf.io/5a6bd/)

Beta coefficients (4D) and native masks for each participant have their own zip file. Participant numbering reflects participants that were excluded in the original studies. Class labels in PyMVPA format are included with each participant (GS study, raw fMRI data from original study at https://osf.io/62rgs/) or in the all_attr.zip file (NI study, raw fMRI data from original study at https://osf.io/qp54f/). The masks.zip file in each study's folder are the masks in MNI space. The mixed models were run in R and use the "reduced" version of the ROI results. The reduced csv files are just averaged across crossvalidation runs. Searchlight results are also included as compressed zip files for each study; formatted as .npy (Numpy) files after decompression.

2) Code on GitHub (https://github.com/bobaseb/neural_sim_measures)

Each study folder (GS_study and NI_study) contains three scripts: one script for performing the neural similarity analysis per region of interest, one for performing the searchlight analysis, and one for performing the mixed effects model. The most commented script is the neural_sim_per_roi_NI.py script in the NI_study folder, so please look at that first to get a basic understanding. Installing PyMVPA (http://www.pymvpa.org/) should get you all the requirements you need to run the scripts. The mixed effects model is written in R and requires the lme4 (https://cran.r-project.org/web/packages/lme4/index.html) and multcomp (https://cran.r-project.org/web/packages/multcomp/index.html) packages.

## Demo:

1) neural_sim_per_roi_\*.py

To test the script on an average computer, try running a single crossvalidation fold for one subject and one region of interest. Depending on your setup this could take a long time, especially for the NS data. Try reducing the number of similarity measures evaluated too if the goal is just testing the code runs properly on your local setup. There are two main functions in this script: clf_wrapper (classification analysis) and dist_wrapper (neural similarity analysis). Expected output for clf_wrapper are the test accuracies, the validation accuracies, the number of voxels selected and the run (chunk) that was used for validation.

2) neural_sim_searchlight_\*.py

This script takes a modified version of clf_wrapper (one of the main functions used in neural_sim_per_roi_\*.py, see above) and provides output for a searchlight analysis using PyMVPA's sphere_searchlight function. This modified version of clf_wrapper does not do optimization for feature selection (picking the top voxels) since this would be intractable even on a university's computer cluster.

3) mixedmodel_\*.R

This script expects the data output from neural_sim_per_roi_\*.py to be averaged across crossvalidation folds. The expected data structure can be viewed in roi_results_\*_reduced.csv.

## Reproduction instructions:

1) Table 1

The data for Table 1 can be constructed from the output of mixedmodel_\*.R

2) Figure 3

The mean Spearman correlations for the similarity measures (i.e., similarity profiles) can be extracted from roi_results_\*.csv to reproduce Figure 3 in the manuscript. 

3) Figure 4 & Figure in Supplemental Information

This requires the output from the searchlight analysis (neural_sim_searchlight_\*.py) to be transformed to MNI space. The appropriate statistical tests can then be run with FSL's randomise function.
