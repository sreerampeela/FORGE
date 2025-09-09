
# FORGE - Factorization Of Response and Gene Essentiality

![Model Architecture](FORGE.png)

FORGE co-models drug response and target gene essentiality, enabling the stratification 
of promising treatment groups for targeted therapy consideration. 

### FORGE's key applications
- Provide BenefitScore to estimate treatment efficacy from basal gene expression profiles.
- Identify putative influencer genes driving treatment response

#### Dependencies
1. Python>=3.10
2. Numpy
3. Pandas
4. Scikit-learn

### Running the pipeline

The full pipeline can be run using 'forge_main.py' script. 

##### Quick start
python forge_main.py \
    --dependency ./test_data/egfr_dependency.csv \
    --ic50 ./test_data/erlotinib_ic50.csv \
    --expression ./test_data/gene_exp.csv \
    --base_name "Erlotinib_EGFR" \
    --target_gene "EGFR"
    
Please run forge_main.py --help for full help menu.

##### Arguments
  - Expression data: Gene expression data with samples as rows and genes as columns. The script **doesnot** determine the highly correlated genes.
  - Dependency: Full/target-specific dependency data.
  - Drug IC50: IC50 values for key drugs from CREAMMIST database
  - Target gene: Name of the target gene in the dependency dataset

### Additional Scripts

The following scripts have been used for dataset preprocessing/downloading and analysis as per the manuscript:

1. limma_voom_general.R - an automated R pipeline to compute Voom-transformed data from raw gene counts
2. getCbioData.R - a script to download gene expression and study ids for selected patients
3. data_Preprocessing.ipynb - a Jupyter Notebook outlining the data cleaning and integration for
   building the FORGE model
4. PDX_metadata_analysis.ipynb - Jupyter Notebook outlining data cleaning and prelimimary analysis for PDX data
5. PDX_data_analysis.ipynb - Jupyter Notebook outlining benefit score computations and comparisons
6. Tahoe_complete_analysis.ipynb - Jupyter Notebook outlining data cleaning and analysis for Tahoe-100M data
7. tahoe_dmso_score_DEG.R - the pipeline from raw pseudobulk counts to DEG analysis for DMSO-treated cell lines
   in Tahoe-100M dataset 



