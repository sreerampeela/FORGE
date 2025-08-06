
# FORGE - Factorization Of Response and Gene Essentiality

![Model Architecture](FORGE.png)

FORGE co-models drug response and target gene essentiality, enabling the stratification 
of promising treatment groups for targeted therapy consideration. 

### FORGE's key applications
- Provide BenefitScore to estimate treatment efficacy from basal gene expression profiles.
- Identify putative influencer genes driving treatment response


### Running the pipeline

The full pipeline can be run using 'forge_main.py' script. 

##### Quick start
python forge_main.py \
    --dependency /path/to/dependency_full_cleaned.csv \
    --ic50 /path/to/erlotinib_ic50.csv \
    --expression /path/to/Erlotinib_expression.csv \
    --base_name "Erlotinib_EGFR" \
    --target_gene "EGFR"
    
Please run forge_main.py --help for full help menu.

##### Arguments
  - Expression data: Gene expression data with samples as rows and genes as columns
  - Dependency: Full/target-specific dependency data
  - Drug IC50: IC50 values for key drugs from CREAMMIST database
  - Target gene: Name of the target gene in the dependency dataset

### Additional Scripts

The following scripts for used for dataset preprocessing/downloading and analysis:

1. limma_voom_general.R - an automated R pipeline to compute Voom-transformed data from raw gene counts
2. getCbioData.R - a script to download gene expression and study ids for selected patients
3. data_Preprocessing.ipynb - a Jupyter Notebook outlining the data cleaning and integration for
   building the FORGE model
4. PDX_analysis.ipynb - Jupyter Notebook outlining data cleaning and analysis for PDX data
5. Tahoe_analysis.ipynb - Jupyter Notebook outlining data cleaning and analysis for Tahoe-100M data
6. tahoe_dmso_score_DEG.R - the pipeline from raw pseudobulk counts to DEG analysis for DMSO-treated cell lines
   in Tahoe-100M dataset 



