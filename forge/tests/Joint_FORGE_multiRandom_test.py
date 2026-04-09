import time
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from JointFORGE import *
from numthreads import set_num_threads
from pyinstrument import Profiler
from memory_profiler import memory_usage
import datetime
import gc

# set max number of threads to 4 for the code here
set_num_threads(4)

exp_path = "./test_data/exp_test.csv"
dep_path = "./test_data/dep_test.csv"
ic50_path = "./test_data/ic50_test.csv"

drug_target_data = pd.read_csv('./test_data/Drug_target_data.csv',
                               header=0, index_col=0)
# drug_target_data.head()
key_drugs = ['ERLOTINIB']
seed_instances = [42, 586231, 321456, 227745, 228796, 12587, 23698, 11111222]
dep_data_full = pd.read_csv(dep_path, header = 0, index_col = 0)
dep_data_genes = dep_data_full.columns.tolist()

exp_data = pd.read_csv(exp_path, header = 0, index_col = 0)
exp_data_genes = exp_data.columns.tolist()

# dep_data_full.head()

#### Run the full pipeline for all drug-target pairs
# with logging file output
model_out_dir = './Models/multiSeed_optuna_indModels'
for rand_seed in seed_instances:
  for drug in key_drugs:
    target_genes = drug_target_data.loc[drug, 'Target'].split(',')
    print(f'Drug {drug}: number of targets = {len(target_genes)}')
    for target in target_genes:
      model_path = os.path.join(model_out_dir, f'{drug}_{target}_forgeModel_optuna100_{rand_seed}.pkl')
      if os.path.exists(model_path):
        print(f'Model for {drug}-{target} pair already exists. Skipping it..')
        continue
       
      elif (target in dep_data_genes) & (target in exp_data_genes) :
        print(f'Building FORGE for {drug}-{target} pair..')
        log_file = os.path.join(model_out_dir, f'{drug}_{target}_logfile_{rand_seed}.txt')
        print(f'Logging in: {log_file}.')
      #### read the intermediate data file to get train and test cell lines ###
        t1 = FORGE(exp_path=exp_path, dependency_path=dep_path, ic50_path=ic50_path,
               drug_name=drug, target_name=target, log_file=log_file, overwrite=True)
        print(f'Running FORGE for {drug}-{target} pair (Optuna tuning included) using seed: {rand_seed}..')
             
        t1.run_Pipeline(n_splits=5, seed_val=rand_seed, tuning_epochs=500,
                                             training_epochs=1000, 
                                             model_path=model_path,
                                             optuna_trials=100)
        
            
      else:
        print(f'Target {target} dependency or expression data not available. skipping it.!')
        continue
  

