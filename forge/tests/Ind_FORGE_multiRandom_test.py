import time
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from JointFORGE import *
from IndFORGE import *
from numthreads import set_num_threads
from pyinstrument import Profiler
from memory_profiler import memory_usage
import datetime
import gc

# set max number of threads to 4 for the code here
set_num_threads(4)

exp_path = "/home/nilabjab/cancer_dependency_project_nilabja/cancer_dependency_project/Approach3_Latent_factor/Fresh_FORGE/Data/Exp.csv"
dep_path = "/home/nilabjab/cancer_dependency_project_nilabja/cancer_dependency_project/Approach3_Latent_factor/Fresh_FORGE/Data/Dep.csv"
ic50_path = "/home/nilabjab/cancer_dependency_project_nilabja/cancer_dependency_project/Approach3_Latent_factor/Fresh_FORGE/Data/Creammist_common_ic50.csv"

drug_target_data = pd.read_csv('/home/sreeramp/cancer_dependency_project/nilabja/Approach3_Latent_factor/Fresh_FORGE/Data/Drug_target_data.csv',
                               header=0, index_col=0)
# drug_target_data.head()
key_drugs = ['AZD8931']
seed_instances = [42, 586231, 321456, 227745, 228796, 12587, 23698, 11111222]
# seed_instances = [161287]
# with open('low_sampleSize_drugs.txt', 'r') as f:
#   ss_low = [i.rstrip().strip() for i in f.readlines()]

# with open('high_sampleSize_drugs.txt', 'r') as f:
#   ss_high = [i.rstrip().strip() for i in f.readlines()]
# key_drugs = ['AZD8931', 'ERLOTINIB', 'IMATINIB', 'MK-2206', 'TIVANTINIB', 'ULIXERTINIB',
#              'UPROSERTIB', 'BMS-754807', 'DABRAFENIB', 'DAPORINAD']
# key_drugs = ss_low[:]
# key_drugs.extend(ss_high)
dep_data_full = pd.read_csv(dep_path, header=0, index_col=0)
dep_data_genes = dep_data_full.columns.tolist()

exp_data = pd.read_csv(exp_path, header=0, index_col=0)
exp_data_genes = exp_data.columns.tolist()

# dep_data_full.head()
optuna_models_path = '/home/sreeramp/cancer_dependency_project/nilabja/Approach3_Latent_factor/git_repo/Models/multiSeed_optuna'
# Run the full pipeline for all drug-target pairs
# with logging file output
model_out_dir = '/home/sreeramp/cancer_dependency_project/nilabja/Approach3_Latent_factor/git_repo/Models/multiSeed_optuna_indModels'
for rand_seed in seed_instances:
    for drug in key_drugs:
        target_genes = drug_target_data.loc[drug, 'Target'].split(',')
        print(f'Drug {drug}: number of targets = {len(target_genes)}')
        for target in target_genes:
            model_path = os.path.join(
                model_out_dir, f'{drug}_{target}_independentModel_optuna_{rand_seed}.pkl')
            if (target in dep_data_genes) & (target in exp_data_genes):
                print(
                    f'Building Individual FORGE for {drug} - {target} pair..')
                log_file = os.path.join(model_out_dir, f'{drug}_{target}_independentModel_logfile_{rand_seed}.txt')
                print(f'Logging in: {log_file}.')
            #### read the intermediate data file to get train and test cell lines ###

                joint_forge_path = os.path.join(
                    optuna_models_path, f'{drug}_{target}_forgeModel_optuna100_{rand_seed}.pkl')
                if not os.path.exists(joint_forge_path):
                    print(
                        f'FORGE JOINT model missing for {drug}-{target} pair..')
                    continue
                else:
                    joint_model = FORGE.load_forge(path=joint_forge_path)
                    print(f'Extracting data from Joint FORGE model..')
                    train_cellLines, test_cellLines, hcg_list = joint_model.train_cellLines, joint_model.test_cellLines, joint_model.hcg_list
                    print(
                        f'train cell lines: {len(train_cellLines)}\ntest cell lines: {len(test_cellLines)}\nHCGs: {len(hcg_list)}')
                    ind_model = IndividualFORGE(exp_path=exp_path, dep_path=dep_path, ic50_path=ic50_path,
                                                hcg_list=hcg_list, train_ids=train_cellLines, test_ids=test_cellLines,
                                                log_file=log_file, overwrite=True, drug=drug, target=target)
                    print(
                        f'Running Individual FORGE for {drug} - {target} pair (Optuna tuning included)..')
            # --- Track memory usage during run_Pipeline ---
                    ind_model.runPipeline(n_splits=5, seed_val=rand_seed,
                                          model_path=model_path,
                                          tuning_epochs=500,
                                          training_epochs=1000, optuna_trials=100)

            else:
                print(
                f'Target {target} dependency or expression data not available. skipping it.!')
                continue
