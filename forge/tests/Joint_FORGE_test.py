import time
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from JointFORGE import *
from numthreads import set_num_threads
import gc

# set max number of threads to 4 for the code here
set_num_threads(4)

exp_path = "./Data/Exp.csv"
dep_path = "./Data/Dep.csv"
ic50_path = "./Data/Creammist_common_ic50.csv"

drug_target_data = pd.read_csv('./Data/Drug_target_data.csv',
                               header=0, index_col=0)
# drug_target_data.head()
key_drugs = ['ERLOTINIB']
dep_data_full = pd.read_csv(dep_path, header=0, index_col=0)
dep_data_genes = dep_data_full.columns.tolist()

exp_data = pd.read_csv(exp_path, header=0, index_col=0)
exp_data_genes = exp_data.columns.tolist()

del dep_data_full, exp_data
gc.collect()
# dep_data_full.head()
optuna_trials = 100
# Run the full pipeline for all drug-target pairs
# with logging file output
optuna_outdir = './Models/optuna_models'
for drug in ss_low:
    target_genes = drug_target_data.loc[drug, 'Target'].split(',')
    print(f'Drug {drug}: number of targets = {len(target_genes)}')
    for target in target_genes:
        if os.path.exists(os.path.join(optuna_outdir, f'{drug}_{target}_forgeModel_optuna100.pkl')):
            print(
                f'Model for {drug}-{target} pair already exists. Skipping it..')
            continue
        else:
            model_path = os.path.join(
                optuna_outdir, f'{drug}_{target}_forgeModel_optuna100.pkl')
        if (target in dep_data_genes) & (target in exp_data_genes):
            print(f'Building FORGE for {drug}-{target} pair..')
            log_file = os.path.join(
                optuna_outdir, f'{drug}_{target}_logfile_optuna_24122025.txt')
            print(f'Logging in: {log_file}.')
            #### read the intermediate data file to get train and test cell lines ###
            t1 = FORGE(exp_path=exp_path, dependency_path=dep_path, ic50_path=ic50_path,
                       drug_name=drug, target_name=target, log_file=log_file, overwrite=True)
            print(
                f'Running FORGE for {drug}-{target} pair (Optuna tuning included)..')
            # --- Track memory usage during run_Pipeline ---
            print(
                f'Warning: expected number of trials = 1200. Using only {optuna_trials} trials..')
            t1.run_Pipeline(n_splits=5, seed_val=198716, tuning_epochs=500,
                            training_epochs=1000,
                            model_path=model_path,
                            optuna_trials=optuna_trials)

        else:
            print(
                f'Target {target} dependency or expression data not available. skipping it.!')
            continue
