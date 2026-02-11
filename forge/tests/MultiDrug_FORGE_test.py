import time
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from MultiDrugFORGE import *
from numthreads import set_num_threads
import gc

# set max number of threads to 4 for the code here
set_num_threads(4)

exp_path = "./test_data/exp_test.csv"
dep_path = "./test_data/dep_test.csv"
ic50_path = "./test_data/ic50_test.csv"

drug_target_data = pd.read_csv('./test_data/Drug_target_data.csv',
                               header=0, index_col=0)
target_genes = ['EGFR']
drug_names = ['ERLOTINIB', 'GEFITINIB']
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
for target in target_genes:

    if (target in dep_data_genes) & (target in exp_data_genes):
        print(f'Building multi drug FORGE for {target}..')
        model_path = os.path.join(
            optuna_outdir, f'{target}_multiDrugFORGE_optuna100.pkl')
        log_file = os.path.join(
            optuna_outdir, f'{target}_MDFORGE_logfile_optuna_04012025.txt')
        print(f'Logging in: {log_file}.')
        #### read the intermediate data file to get train and test cell lines ###
        t1 = MultiDrugFORGE(exp_path=exp_path, dependency_path=dep_path, ic50_path=ic50_path, drug_names=drug_names,
                            drug_target_data=drug_target_path, target_name=target, log_file=log_file, overwrite=True)
        print(
            f'Running multi drug FORGE for {target} pair (Optuna tuning included)..')
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
