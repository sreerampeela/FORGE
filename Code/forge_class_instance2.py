from base64 import b16decode
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import random
import pickle
import optuna
import time


class FORGE:
    '''
    Main class object of the FORGE framework.

The FORGE class takes three biological matrices as input—

1. Gene expression matrix (typically restricted to a subset of highly correlated genes),
2. Gene dependency scores, and
3. Drug IC50 profiles

and converts all of them into standardized pandas DataFrame objects.

Once initialized, FORGE performs multiple training iterations. In each
iteration, the latent space weight matrix (W) is randomly initialized,
and the model is optimized using a specified loss function. A full
hyper-parameter search is then performed (e.g., using Optuna) to
identify the optimal combination of latent dimension, learning rate,
L1-penalty, and task-weighting parameter.

For every random initialization and hyper-parameter configuration,
the model produces three learnable matrices:
    • W  –  latent gene weights
    • hD – dependency prediction head
    • hI – IC50 prediction head

After running all iterations, FORGE considers the instance with minimal
Validation loss as the best model, and is then saved as W,hD and hI for
for the model class. The total results are also stored as a pkl file for
later use.

This enables robust inference of gene-level latent features and drug
response relationships that are less sensitive to random initialization
or hyper-parameter noise.

    '''

    def __init__(self, exp_path, dependency_path, ic50_path, drug_name, target_name, hcg_list):
        '''
        Reads in paths for all the three datasets - gene expression, target dependency,
        and drug IC50, and subsets them to include a common set of cell lines.
        '''
        self.exp_path = exp_path
        self.dep_path = dependency_path
        self.ic50_path = ic50_path
        if not all([os.path.exists(self.exp_path), os.path.exists(self.dep_path), os.path.exists(self.ic50_path)]):
            raise FileNotFoundError(
                f'One of files used for creating the model is missing..Please re-run FORGE with correct paths.')
            sys.exit(1)
        exp_data = pd.read_csv(exp_path, header=0, index_col=0)
        dep_data = pd.read_csv(dependency_path, header=0, index_col=0)
        ic50_data = pd.read_csv(ic50_path, header=0, index_col=0)
        ic50_data = ic50_data.T  # cell lines as rows
        if drug_name not in ic50_data.columns:
            raise ValueError(f'{drug_name} not in the IC50 dataset')
            sys.exit(1)
        if target_name not in dep_data.columns:
            raise ValueError(f'{target_name} not in the Dependency dataset')
            sys.exit(1)
        # clean the data for drug and target-specific entries
        print('Cleaning the input datasets..')
        ic50_data = ic50_data[drug_name].dropna()
        dep_data = dep_data[target_name].dropna()
        # Both are arrays now.
        common_cell_lines = list(set(exp_data.index) & set(
            dep_data.index) & set(ic50_data.index))
        if not common_cell_lines:
            raise ValueError(
                f'No cell lines were common for the {drug_name}-{target_name} pair. Exiting!')
            sys.exit(1)
        else:
            print(
                f'Identified a total of {len(common_cell_lines)} samples in input data')
            self.exp_data = exp_data.loc[common_cell_lines, hcg_list]
            self.dep_data = dep_data[common_cell_lines]
            self.ic50_data = ic50_data[common_cell_lines]
            self.drug = drug_name
            self.target = target_name

    def get_train_val_test_kSplits(self, n_splits=5, test_size=0.2, seed=198716):
        '''
        Performs train-val-test split and returns such dataframes for further analysis.
        The test data is outside of these splits
        '''
        num_samples = self.exp_data.shape[0]

        print(f"Performing k-fold split over {num_samples} samples...")

# -------------------------------------------------------
# 1. Shuffle all indices (positions 0..N-1)
# -------------------------------------------------------
        rng = np.random.default_rng(seed)
        all_idx = np.arange(num_samples)
        rng.shuffle(all_idx)

        # -------------------------------------------------------
        # 2. 20% test split
        # -------------------------------------------------------
        n_test = int(num_samples * test_size)
        test_pos = all_idx[:n_test]                   # integer positions
        test_cellLines = self.exp_data.index[test_pos]
        self.test_cellLines = test_cellLines

        # Remaining 80% training positions
        train_pos_full = all_idx[n_test:]

        # -------------------------------------------------------
        # 3. 5-fold CV on remaining 80%
        # -------------------------------------------------------
        kf = KFold(n_splits=n_splits, shuffle=False)

        self.training_cv_splits = []

        for train_ids, val_ids in kf.split(train_pos_full):
            train_pos = train_pos_full[train_ids]     # positions
            val_pos = train_pos_full[val_ids]

            train_cellLines = self.exp_data.index[train_pos]
            val_cellLines = self.exp_data.index[val_pos]

            self.training_cv_splits.append((train_cellLines, val_cellLines))

    def initialise_model(self, seed_val, G, K):
        np.random.seed(seed_val)
        W = np.random.rand(G.shape[1], K)
        hD = np.random.rand(K, 1)
        hI = np.random.rand(K, 1)
        return W, hD, hI

    def train_instance(self, G_train, D_train, I_train,
                       G_val, D_val, I_val,
                       W, lmbda, lr, hD, hI,
                       num_epochs=5000, rand_seed=198716):
        '''
        For n epochs, and a set of hyper-params, learn the matrices
        '''
        train_losses = []
        val_losses = []
        G_train = np.asarray(G_train)       # shape (n_samples, n_genes)
        G_val = np.asarray(G_val)
        hD = np.asarray(hD).reshape(-1, 1)     # shape (k, 1) or (k,)
        hI = np.asarray(hI).reshape(-1, 1)

        for epoch in range(num_epochs):
            Z_train = G_train @ W
            multi_matrix = np.linalg.pinv(Z_train.T @ Z_train) @ Z_train.T
            hD = multi_matrix @ D_train
            hI = multi_matrix @ I_train
            hD = np.asarray(hD).reshape(-1, 1)   # (20,1)
            hI = np.asarray(hI).reshape(-1, 1)

            # print('Latent vectors dimensions:', hD.shape, hI.shape)
            train_pred_d = Z_train @ hD
            train_pred_i = Z_train @ hI

            train_err_d = train_pred_d - D_train
            train_err_i = train_pred_i - I_train
            # print('Error vectors dimensions:', err_d.shape, err_i.shape)
            train_err_d = np.asarray(train_err_d).reshape(-1, 1)
            train_err_i = np.asarray(train_err_i).reshape(-1, 1)
            # print('Error vectors dimensions after reshaping:',
            #   err_d.shape, err_i.shape)
            train_loss = (np.mean(train_err_d**2) + np.mean(train_err_i ** 2)
                          ) * 0.5  # average of both MSEs

            grad_d = 2 * G_train.T @ (train_err_d @ hD.T)
            grad_i = 2 * G_train.T @ (train_err_i @ hI.T)

            grad_W = 2 * (grad_d + grad_i) + lmbda * np.sign(W)
            W -= lr * grad_W
            val_loss = self.validate_instance(G_val=G_val, D_val=D_val, I_val=I_val,
                                              W=W, hD=hD, hI=hI)
            val_losses.append(val_loss)
            if (epoch + 1) % 100 == 0:

                print(
                    f'Epoch {epoch + 1}: Training loss:\n  Dependency: {np.mean(train_err_d):.5f}\n   IC50: {np.mean(train_err_i):.5f}')
            train_losses.append(round(train_loss, 5))

        return W, hD, hI, train_losses, val_losses

    def validate_instance(self, G_val, D_val, I_val, W, hD, hI):
        '''
        For n epochs, and a set of hyper-params, learn the matrices
        '''
        # val_losses = []
        # val_losses = []
        G_val = np.asarray(G_val)       # shape (n_samples, n_genes)
        D_val = np.array(D_val).reshape(-1, 1)
        I_val = np.array(I_val).reshape(-1, 1)
        hD = np.asarray(hD).reshape(-1, 1)     # shape (k, 1) or (k,)
        hI = np.asarray(hI).reshape(-1, 1)
        Z = G_val @ W

        pred_d = Z @ hD
        pred_i = Z @ hI

        err_d = pred_d - D_val
        err_i = pred_i - I_val
        # print('Error vectors dimensions:', err_d.shape, err_i.shape)
        err_d = np.asarray(err_d).reshape(-1, 1)
        err_i = np.asarray(err_i).reshape(-1, 1)
        # print('Error vectors dimensions after reshaping:',
        #   err_d.shape, err_i.shape)
        val_loss = (np.mean(err_d**2) + np.mean(err_i ** 2)
                    ) * 0.5  # average of both MSEs

        # val_losses.append(round(val_loss, 5))
        return val_loss

    def optuna_tuner(self, G_train, D_train, I_train, G_val, D_val, I_val, seed_val, num_epochs=1000, n_trials=5):
        """
        Hyperparameter tuning using Optuna based on validation loss.
        """
        instance_name = f'{self.drug}-{self.target}_optuna_{seed_val}'
        print(
            f"\n--- Starting Optuna search for {self.drug} - {self.target} ---")

    # --------------------------
    # Objective function
    # --------------------------
        def objective(trial):
            np.random.seed(seed_val)

        # Hyperparameters
            K = trial.suggest_categorical("latent_dim", [10, 20, 30, 40, 50])
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            lmbda = trial.suggest_float("lambda", 1e-4, 1e-1, log=True)

            # Initialize W, hD, hI
            W = np.random.randn(G_train.shape[1], K)
            hD = np.random.randn(K, 1)
            hI = np.random.randn(K, 1)

        # --------------------------
        # Mini training loop (short)
        # --------------------------
            tuned_W, tuned_hD, tuned_hI, train_losses, val_losses = self.train_instance(G_train=G_train, D_train=D_train,
                                                                                        I_train=I_train, G_val=G_val,
                                                                                        D_val=D_val, I_val=I_val,
                                                                                        W=W, lmbda=lmbda, lr=lr, hD=hD, hI=hI,
                                                                                        num_epochs=num_epochs, rand_seed=seed_val)

            self.__setattr__(f'train_loss_trial{trial.number}', train_losses)
            self.__setattr__(f'val_loss_trial{trial.number}', val_losses)

        # --------------------------
        # VALIDATION LOSS
        # --------------------------
            val_loss = self.validate_instance(G_val=G_val, D_val=D_val,
                                              I_val=I_val, W=tuned_W, hD=tuned_hD, hI=tuned_hI)
            print(
                f'Validation loss at end of trial {trial.number}: {val_loss:.5f}')
            return val_loss    # Optuna minimizes this

    # --------------------------
    # Create study
    # --------------------------
        study = optuna.create_study(
            study_name=instance_name,
            direction="minimize",
            storage=f"sqlite:///{instance_name}.db",
            load_if_exists=True
        )
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)  # for testing
        end_time = time.time()
        print(
            f'Hyper-param tuning completed in {end_time - start_time} seconds..')
        # print("\nBest params:", study.best_params)

        return study.best_params

    def train_forge(self, fold_name, train_ids, val_ids, optuna_trials=100,
                    tuning_epochs=500, training_epochs=5000, seed_val=198716):
        '''
        Main function to learn W, hD and hI matrices using specific number of epochs
        and set number of randomly initialized Ws. The final matrices are aggregated using
        mean of each bootstrap instance.
        '''
        # Ws, hDs, hIs, total_losses = [], [], [], []
        G_train = self.exp_data.loc[train_ids].values  # matrix
        # 1D vector
        D_train = self.dep_data.loc[train_ids].values.reshape(-1, 1)
        # 1D vector
        I_train = self.ic50_data.loc[train_ids].values.reshape(-1, 1)
        G_val = self.exp_data.loc[val_ids].values  # matrix
        D_val = self.dep_data.loc[val_ids].values.reshape(-1, 1)  # 1D vector
        I_val = self.ic50_data.loc[val_ids].values.reshape(-1, 1)
        print('Scaling the matrices...')
        # scale train with its mean, and val with train mean
        self.mean_ic50 = I_train.mean()
        self.mean_dep = D_train.mean()
        I_train = I_train - self.mean_ic50
        D_train = D_train - self.mean_dep
        I_val = I_val - self.mean_ic50
        D_val = D_val - self.mean_dep
        print(
            f'Starting FORGE training with single random instance of W..')
        print(f'Tuning hyper-params..')
        # optuna_tuner(self, G_train, D_train, I_train, G_val, D_val, I_val, seed_val, num_epochs=1000, n_trials=5)
        best_hyperparams = self.optuna_tuner(G_train=G_train, D_train=D_train, I_train=I_train,
                                             G_val=G_val, D_val=D_val, I_val=I_val, n_trials=optuna_trials,
                                             seed_val=seed_val, num_epochs=tuning_epochs)
        print(
            f'Best hyper-params:\n{best_hyperparams}')
        print('Training using the best hyper-params..')
        num_latent = best_hyperparams['latent_dim']
        lr_best = best_hyperparams['learning_rate']
        best_lambda = best_hyperparams['lambda']
        # train_instance(G, D, I, W, lmbda, lr, hD, hI, num_epochs=5000, rand_seed=198716)
        W = np.random.randn(G_train.shape[1], num_latent)
        hD = np.random.randn(num_latent, 1)
        hI = np.random.randn(num_latent, 1)
        # Train using the reshaped matrices and best hyper-params
        final_W, final_hD, final_hI, train_losses_final, val_losses_final = self.train_instance(G_train=G_train, D_train=D_train, I_train=I_train,
                                                                                                G_val=G_val, D_val=D_val, I_val=I_val,
                                                                                                W=W, hD=hD, hI=hI,
                                                                                                rand_seed=seed_val, lmbda=best_lambda,
                                                                                                lr=lr_best, num_epochs=training_epochs)  # train for more epochs
        self.__setattr__(
            f'fold{str(fold_name)}_epoch_train_loss', train_losses_final)
        self.__setattr__(
            f'fold{str(fold_name)}_epoch_val_loss', val_losses_final)
        train_predictions_D = G_train @ final_W @ final_hD
        train_predictions_I = G_train @ final_W @ final_hI
        tuned_train_loss = 0.5 * \
            (((train_predictions_D - D_train) ** 2).mean() +
             ((train_predictions_I - I_train) ** 2).mean())

        val_predictions_D = G_val @ final_W @ final_hD
        val_predictions_I = G_val @ final_W @ final_hI
        tuned_val_loss = 0.5 * \
            (((val_predictions_D - D_val) ** 2).mean() +
             ((val_predictions_I - I_val) ** 2).mean())
        print(
            f"Final training loss for fold {str(fold_name)}: {tuned_train_loss:.5f}. "
            f"Validation loss for fold {str(fold_name)}: {tuned_val_loss:.5f}"
        )

        # assign to the model, the one with least val loss
        setattr(self, f"fold{fold_name}_W", final_W)
        setattr(self, f"fold{fold_name}_hD", final_hD)
        setattr(self, f"fold{fold_name}_hI", final_hI)

        setattr(self, f"fold{fold_name}_train_predictions_dep",
                train_predictions_D)
        setattr(self, f"fold{fold_name}_train_predictions_IC50",
                train_predictions_I)

        setattr(self, f"fold{fold_name}_val_predictions_dep",
                val_predictions_D)
        setattr(
            self, f"fold{fold_name}_val_predictions_IC50", val_predictions_I)

    def report_metrics(self, G, W, hD, hI, actual_dep, actual_ic50):
        pred_d = G @ W @ hD
        pred_I = G @ W @ hI
        pearson_corr1, pearson_p1 = pearsonr(x=actual_dep, y=pred_d)
        spearman_corr1, spearman_p1 = spearmanr(a=actual_dep, b=pred_d)
        pearson_corr2, pearson_p2 = pearsonr(x=actual_ic50, y=pred_I)
        spearman_corr2, spearman_p2 = spearmanr(a=actual_ic50, b=pred_I)
        mse_D = mean_squared_error(actual_dep, pred_d)
        mse_I = mean_squared_error(actual_ic50, pred_I)
        res_dict = {
            'dep_pearson_corr': round(float(pearson_corr1), 5),
            'dep_spearman_corr': round(float(spearman_corr1), 5),
            'ic50_pearson_corr': round(float(pearson_corr2), 5),
            'ic50_spearman_corr': round(float(spearman_corr2), 5),
            'dep_MSE': round(float(mse_D), 5),
            'ic50_MSE': round(float(mse_I), 5)
        }

        return res_dict

    # -----------------------------
    # REMOVE DataFrames before saving
    # -----------------------------
    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove massive DataFrames
        state.pop("exp_data", None)
        state.pop("dep_data", None)
        state.pop("ic50_data", None)
        print(state)  # omit printing the dataframes
        return state

    # -----------------------------
    # RELOAD DataFrames when loading
    # -----------------------------
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not all([os.path.exists(self.exp_path), os.path.exists(self.dep_path), os.path.exists(self.ic50_path)]):
            raise FileNotFoundError(
                f'One of files used for creating the model is missing..Please re-run FORGE with correct paths.')
            sys.exit(1)
        # Use stored paths to reload
        self.exp_data = pd.read_csv(self.exp_path, index_col=0)
        self.dep_data = pd.read_csv(self.dep_path, index_col=0)
        self.ic50_data = pd.read_csv(self.ic50_path, index_col=0)

    # -----------------------------
    # SAVE the whole object
    # -----------------------------
    def save_forge(self, path=None):
        if path is None:
            model_name = f"{self.drug}_{self.target}.pkl"
        else:
            model_name = path

        with open(model_name, "wb") as f:
            pickle.dump(self, f)

        print(f"FORGE model saved: {model_name}")

    # -----------------------------
    # LOAD saved FORGE object
    # -----------------------------
    @staticmethod
    def load_forge(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"FORGE model loaded from: {path}")
        return obj

    def run_Pipeline(self, n_splits, seed_val, tuning_epochs=500,
                     training_epochs=5000, optuna_trials=100, model_path='test_forge.pkl'):
        '''
        A single method to run full pipeline from tuning and training + testing
        '''
        self.get_train_val_test_kSplits(
            test_size=0.2, n_splits=n_splits, seed=seed_val)

        start_time1 = time.time()
        # no stdout for epochs
        # Create a loop for 5-fold
        for i in range(n_splits):
            print('Running for fold:', str(i))
            print(
                f"Experiment setup: "
                f"Train={len(self.training_cv_splits[i][0])}, "
                f"Val={len(self.training_cv_splits[i][1])}, "
                f"Test={len(self.test_cellLines)}"
            )

            train_ids, val_ids = self.training_cv_splits[i]
            self.train_forge(fold_name=i, train_ids=train_ids, val_ids=val_ids, optuna_trials=optuna_trials,
                             tuning_epochs=tuning_epochs,
                             seed_val=seed_val, training_epochs=training_epochs)

            print('Tuning completed for fold:', i)
            end_time1 = time.time()
            print(
                f'FORGE training completed in {end_time1 - start_time1} seconds..Running test..')
            test_ids = self.test_cellLines
            G_test = self.exp_data.loc[test_ids].values

            # 1D vector
            D_test = self.dep_data.loc[test_ids].values.reshape(-1, 1)
        # 1D vector
            I_test = self.ic50_data.loc[test_ids].values.reshape(-1, 1)
        # scaling the dependency and IC50 values
            D_test = D_test - self.mean_dep
            I_test = I_test - self.mean_ic50
            fold_hD = getattr(self, f"fold{str(i)}_hD")
            fold_W = getattr(self, f"fold{str(i)}_W")
            fold_hI = getattr(self, f"fold{str(i)}_hI")
            test_predictions_D = G_test @ fold_W @ fold_hD
            test_predictions_I = G_test @ fold_W @ fold_hI
            tuned_test_loss = 0.5 * \
                (((test_predictions_D - D_test) ** 2).mean() +
                 ((test_predictions_I - I_test) ** 2).mean())
            print(
                f"Final test loss for fold {i}: {tuned_test_loss:.5f}. "
            )
        # report_metrics(G, W, hD, hI, actual_dep, actual_ic50)
            test_metrics = self.report_metrics(G=G_test, W=fold_W, hD=fold_hD,
                                               hI=fold_hI, actual_dep=D_test, actual_ic50=I_test)
            print(test_metrics)
            self.__setattr__(f'Fold{str(i)}_test_res', test_metrics)
        print(
            f'Pipeline successfully completed for {self.drug}-{self.target} pair. Saving FORGE model..')
        self.save_forge(path=model_path)
