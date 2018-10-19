# neural_sim_measures
Code for Measures of Neural Similarity (https://doi.org/10.1101/439893)

Each study folder (GS_study and NI_study) contains three scripts: one script for performing the neural similarity analysis per region of interest, one for performing the searchlight analysis, and one for performing the mixed effects model. The most commented script is the neural_sim_per_roi_NI.py script in the NI_study folder, so please look at that first to get a basic understanding. Installing PyMVPA (http://www.pymvpa.org/) should get you all the requirements you need to run the scripts. The mixed effects model is written in R and requires the lme4 (https://cran.r-project.org/web/packages/lme4/index.html) and multcomp (https://cran.r-project.org/web/packages/multcomp/index.html) packages.
