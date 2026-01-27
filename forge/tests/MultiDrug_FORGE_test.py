import time
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from forge_multidrug import *
from numthreads import set_num_threads
# from pyinstrument import Profiler
# from memory_profiler import memory_usage
# import datetime
import gc

# set max number of threads to 4 for the code here
set_num_threads(4)

exp_path = "/home/nilabjab/cancer_dependency_project_nilabja/cancer_dependency_project/Approach3_Latent_factor/Fresh_FORGE/Data/Exp.csv"
dep_path = "/home/nilabjab/cancer_dependency_project_nilabja/cancer_dependency_project/Approach3_Latent_factor/Fresh_FORGE/Data/Dep.csv"
ic50_path = "/home/nilabjab/cancer_dependency_project_nilabja/cancer_dependency_project/Approach3_Latent_factor/Fresh_FORGE/Data/Creammist_common_ic50.csv"

drug_target_path = '/home/sreeramp/cancer_dependency_project/nilabja/Approach3_Latent_factor/Fresh_FORGE/Data/Drug_target_data.csv'
target_genes = ['EGFR']
drug_names = ['ERLOTINIB', 'GEFITINIB']
# # drug_target_data.head()
# # key_drugs = ['ERLOTINIB']
# with open('low_sampleSize_drugs.txt', 'r') as f:
#     ss_low = [i.rstrip().strip() for i in f.readlines()]

# with open('high_sampleSize_drugs.txt', 'r') as f:
#     ss_high = [i.rstrip().strip() for i in f.readlines()]
# # key_drugs = ['ERLOTINIB','DAPORINAD', 'IMATINIB', 'MK-2206', 'TIVANTINIB', 'ULIXERTINIB',
# #                 'UPROSERTIB', 'BMS-754807', 'DABRAFENIB']
# # key_drugs.extend(s
# ss_low.extend(ss_high)
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
optuna_outdir = '/home/sreeramp/cancer_dependency_project/nilabja/Approach3_Latent_factor/git_repo/Models/optuna_models'
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
        # --- Track memory usage during run_Pipeline ---
        # def run_Pipeline(self, n_splits, seed_val, tuning_epochs=500, quiet=False,
        #            training_epochs=5000, optuna_trials=100, model_path='test_forge.pkl'):
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
