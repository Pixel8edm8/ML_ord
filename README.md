The source code is an implementation of our method described in the paper "Isabelle Bichindaritz1, Guanghui Liu1, and Christopher Bartlett1. Integrative Survival Analysis of Breast Cancer with Gene Expression and DNA Methylation Data". 

Because of the github limit that the uploaded file size cannot exceed 100Mb, two original mRNA and methylation datasets failed to upload to the github repository. The whole source code for the proposed method and the datasets have been made available at https://pan.baidu.com/s/1jQP5e-EBe0FOhvf7s5BIkw (extracted code is: LGH0) for free academic use.

## Dependencies

Before running this code, you will need to install ```R``` and ```python```. In our experiment, ```R3.62``` and ```python3.6``` or more advanced versions were tested. This code was tested in WIN7/win10 64 Bit. It should be able to run in other Linux or Windows systems.

## How to run
1. In R, in folder: ```\ML_ordCOX\mRNA_methylation_merged\mRNA data```, run ```run_test_lmQCM_mRNA.r```. Similarly, in folder: ```\ML_ordCOX\mRNA_methylation_merged\methylation data\```, run ```run_test_lmQCM_brca_methylation.r```. If the code runs successfully, the extracted mRNA features and methylation features will be obtained using lmQCM method, respectively. Run ```methylation-mRNA_all_lmQCM_merge.r``` and combine the mRNA and methylation features in sequence. You will obtain a 133-dimensional feature vector which will be viewed as an integrated gene feature input.
2. After obtaining the specific feature representation in step 1, copy the integrated features to folder ```\ML_ordCOX\survival analysis\```. In Python, run ```brca_methylation_mRNA_lmqcm.py``` in folder ```\ML_ordCOX\survival analysis\```.

## --Output
If the code runs successfully, the results will be placed in ```\LSTM-COX-CODE\survival analysis\c_indices.npy```.

Email: Guanghui.liu@oswego.edu.