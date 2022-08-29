# PerioML
Code for the paper on comparing machine learning algorithms for predictive modelling of periodontitis.

#### Authors
Nasir Zeeshan Bashir, Zahid Rahman, Sam Li-Sheng Chen

#### Links
[Published paper](https://onlinelibrary.wiley.com/doi/10.1111/jcpe.13692) (doi: 10.1111/jcpe.13692)

### Background
This Python code was used to develop the compare the validity of various machine learning algorithms in the development and validation of predictive models for periodontitis. The data used were from two cross-sectional studies, one carried out in Taiwan and one in the United States. The findings were published in the Journal of Clinical Periodontology.

### Abstract
**Aim:** The aim of this study was to compare the validity of different machine learning algorithms to develop and validate predictive models for periodontitis. \
**Materials and Methods:** Using national survey data from Taiwan (*n* = 3453) and the United States (*n* = 3685), predictors of periodontitis were extracted from the datasets and preprocessed, and then 10 machine learning algorithms were trained to develop predictive models. The models were validated both internally (bootstrap sampling) and externally (alternative country’s dataset). The algorithms were compared across six performance metrics ([i] area under the curve for the receiver operating characteristic [AUC], [ii] accuracy, [iii] sensitivity, [iv] specificity, [v] positive predictive value, [vi] negative predictive value) and two methods of data preprocessing ([i] machine learning-based feature selection, [ii] dimensionality reduction into principal components). \
**Results:** Many algorithms showed extremely strong performance during internal validation (AUC > 0.95, accuracy > 95%). However, this was not replicated in external validation, where predictive performance of all algorithms dropped off drastically. Furthermore, predictive performance differed according to data pre-processing methodology and the cohort on which they were trained. \
**Conclusions:** Larger sample sizes and more complex predictors of periodontitis are required before machine learning can be leveraged to its full potential.

### Analysis
The Python scripts were executed in the following order: (1) libraries.py, (2) functions.py, (3) preproc_taiwan.py, (4) preproc_nhanes.py, (5) tuning.py, (6) validating.py. It is not possible to generically run these on any dataset as much of the model tuning and validation is specific to our data and variables. These scripts should be adjusted and the hyperparameters adjusted appropriately for your analysis purposes.

**Python v3.9.7**
