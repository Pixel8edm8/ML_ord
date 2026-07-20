# Introduction

The source code is an implementation of our method described in the paper "Isabelle Bichindaritz1, Guanghui Liu1, and Christopher Bartlett1. Integrative Survival Analysis of Breast Cancer with Gene Expression and DNA Methylation Data".

## Dependencies

Before running this code, you will need to install ```R``` and ```python```. In our experiment, ```R3.62``` and ```python3.6``` or more advanced versions were tested. This code was tested in WIN7/win10 64 Bit. It should be able to run in other Linux or Windows systems.

## How to run

1. In R, in folder: ```\ML_ordCOX\mRNA_methylation_merged\mRNA data```, run ```run_test_lmQCM_mRNA.R``` (which will generate ```eigengene_matrix_mRNAseq_RPKM_minClusterSize10.csv``` in the same folder).  Similarly, in folder: ```\ML_ordCOX\mRNA_methylation_merged\methylation data\```, run ```run_test_lmQCM_brca_methylation.R``` (which will generate both ```expr_brca_mRNA.csv``` and ```eigengene_matrix_methylation10.csv``` in the same folder). If the code runs successfully, the extracted mRNA features and methylation features will be obtained using lmQCM method, respectively. Run ```methylation-mRNA_all_lmQCM_merge.R``` and combine the mRNA and methylation features in sequence. You will obtain a 133-dimensional feature vector which will be viewed as an integrated gene feature input.
2. After obtaining the specific feature representation in step 1, copy the integrated features to folder ```\ML_ordCOX\survival analysis\```. In Python, run ```brca_methylation_mRNA_lmqcm.py``` in folder ```\ML_ordCOX\survival analysis\```.

## Output

If the code runs successfully, the results will be placed in ```\LSTM-COX-CODE\survival analysis\c_indices.npy```.

Email: <Guanghui.liu@oswego.edu>.

## On The Docker Container and Development

If you would like to develop this software, you will want to use the Docker Container provided. Since large amounts of this codebase is deprecated, a very specific development environment must be specified to ensure correct running, development, and deployment of this software.

This software utilizes ```conda``` and ```renv``` in conjunction with ```docker``` to fabricate a reproducible environment. Please read the documentation of conda and renv, so that you may have an operational understanding of how to develop with these pieces of software.