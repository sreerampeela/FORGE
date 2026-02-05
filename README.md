
# FORGE - Factorization Of Response and Gene Essentiality

![Model Architecture](FORGE.png)

FORGE co-models drug response and target gene essentiality, enabling the stratification 
of promising treatment groups for targeted therapy consideration. 

### FORGE's key applications
- Provide BenefitScore to estimate treatment efficacy from basal gene expression profiles.
- Identify putative influencer genes driving treatment response

#### Dependencies
1. Python>=3.10
2. Numpy = 2.2.5
3. Pandas = 2.2.3
4. Scikit-learn = 1.6.1

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

### Main Scripts

The following scripts have been used for dataset analysis as per the manuscript:

1. data_Preprocessing.ipynb - a Jupyter Notebook outlining the data cleaning and integration for
   building the FORGE model
2. PDX_metadata_analysis.ipynb - Jupyter Notebook outlining data cleaning and prelimimary analysis for PDX data
3. pdx_data_analysis.ipynb - Jupyter Notebook outlining benefit score computations and comparisons
4. tahoe_complete_analysis.ipynb - Jupyter Notebook outlining data cleaning and analysis for Tahoe-100M data
5. tahoe_deg_analysis.ipynb - the pipeline from raw pseudobulk counts to DEG analysis for DMSO-treated cell lines
   in Tahoe-100M dataset
6. tahoe_dmso_pseudobulk.ipynb - pseudobulk generation for plate 9 alone for the Tahoe-100M dataset
7. depMap_benefitScore_DEG.ipynb - DEG analysis for the DepMap data based on benefit score categories
8. depMap_keyCluster_deg.ipynb - DEG analysis for the key susceptible clusters in the DepMap dataset
9. model_building.ipynb - model building steps for the EGFR-Erlotinib and NAMPT-Daporinad pairs with intermittent analysis


### Supplementary scripts

The following scripts are used in data generation/pre-processing:

1. create_PDX_ExpData.py - script to generate pseudobulk from FPKM values for the PDX dataset
2. limma_voom_general.R - a customized script to run Limma's voom normalisation
3. limma_voom_tahoe.R - customized script to run Limma's voom on Tahoe-100M data (can specify drug name)
4. tahoe_downloader.py - script to automate the downloads for Tahoe-100M plates 7-10 and generate a merged annData file, and plate-specific pseudobulk data.



