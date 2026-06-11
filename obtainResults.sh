#!/bin/bash
Rscript -e "renv::restore()"
Rscript ./mRNA_methylation_merged/mRNA\ data/run_test_lmQCM_mRNA.R
Rscript ./mRNA_methylation_merged/methylation_data/run_test_lmQCM_brca_methylation.R
# Then, copy the merged stuff from the methylation_merged folder into survival_analysis.
cp 