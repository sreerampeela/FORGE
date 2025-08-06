
# FORGE - Factorization Of Response and Gene Essentiality

![Model Architecture](FORGE.png)

FORGE co-models drug response and target gene essentiality, enabling the stratification 
of promising treatment groups for targeted therapy consideration. 

### FORGE's key applications
- Provide BenefitScore to estimate treatment efficacy from basal gene expression profiles.
- Identify putative influencer genes driving treatment response


### Running the pipeline

The full pipeline can be run using 'forge_main.py' script. Please run forge_main.py --help for full help menu.

Arguments:
  - Expression data: Gene expression data with samples as rows and genes as columns
  - Dependency: Full/target-specific dependency data
  - Drug IC50: IC50 values for key drugs from CREAMMIST database
  - Target gene: Name of the target gene in the dependency dataset





