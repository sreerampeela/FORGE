from threadpoolctl import threadpool_limits
from collections import Counter
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, rankdata
import random
import pickle
import time
import logging
from itertools import product
from datetime import datetime
import os
# IMPORTANT: set BLAS thread env vars BEFORE importing heavy numerical libs in worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


timestamp = datetime.now().strftime("%d%m%Y")


def spearman_corr_fun(exp_data, y):
    """
    Compute Spearman correlation between a vector y and each column (gene) in exp_data.

    exp_data : DataFrame (n_samples × n_genes)
    y        : Pandas Series or 1D array-like of length n_samples

    Returns:
        Pandas Series of Spearman correlations for each gene.
    """

    # The exp_data and y should have matching indices

    # ---------------------------------------------------------
    # Rank-transform expression (Spearman = Pearson of ranks)
    # ---------------------------------------------------------
    x_rank = exp_data.apply(rankdata, axis=0)
    x_rank = pd.DataFrame(x_rank, index=exp_data.index,
                          columns=exp_data.columns)

    # Z-transform columns for Pearson correlation
    x_rank = (x_rank - x_rank.mean()) / x_rank.std()

    # ---------------------------------------------------------
    # Rank-transform y (Spearman)
    # ---------------------------------------------------------
    y = y.loc[exp_data.index]  # align y with exp_data indices
    y_rank = rankdata(y.to_numpy())
    y_rank = (y_rank - y_rank.mean()) / y_rank.std()

    # ---------------------------------------------------------
    # Vectorized Pearson correlations
    # (x_rank.T dot y_rank) / (n-1)
    # ---------------------------------------------------------
    n = exp_data.shape[0]
    corr = np.dot(x_rank.to_numpy().T, y_rank) / (n - 1)

    return pd.Series(corr, index=exp_data.columns)


class IndividualFORGE:
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
hyper-parameter search is then performed (using optuna) to
identify the optimal combination of latent dimension, learning rate,
L1-penalty by identifying the combination wit least loss.

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

    def __init__(self, exp_path, response_path, response_name, model_type='IC50', log_file=None, overwrite=False):
        '''
        Reads in paths for all the three datasets - gene expression, target dependency,
        and drug IC50, and subsets them to include a common set of cell lines.
        '''
        if log_file is None:
            log_file = "test_logger.log"

        # 1. Forcefully remove all existing handlers attached to the root logger
        # This effectively "resets" logging so basicConfig will work again.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()
        # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
        if os.path.exists(log_file):
            if overwrite:
                # Overwriting existing log
                logging.basicConfig(
                    filename=log_file,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filemode="w"      # overwrite
                )
                logging.info(
                    "\n****************************************************\n"
                    "*            New Individual FORGE Run Initialised             *\n"
                    "****************************************************"
                )
            else:
                # Appending to existing log
                logging.basicConfig(
                    filename=log_file,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filemode="a"      # append
                )
                logging.info(
                    "\n****************************************************\n"
                    "*        Continuing Previous Individual FORGE Run             *\n"
                    "****************************************************"
                )

        else:
            # Log file does NOT exist → treat as new run
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                filemode="w"
            )
            logging.info(
                "\n****************************************************\n"
                "*            New IndividualFORGE Run Initialised             *\n"
                "****************************************************"
            )
        self.exp_path = exp_path
        self.target_path = response_path
        if not all([os.path.exists(self.exp_path), os.path.exists(self.target_path)]):
            raise FileNotFoundError(
                f'One of files used for creating the model is missing..Please re-run FORGE with correct paths.')

        exp_data = pd.read_csv(exp_path, header=0, index_col=0)
        response_data = pd.read_csv(response_path, header=0, index_col=0)
        if response_name not in response_data.columns:
            response_data = response_data.T  # Cell lines as rows
        if response_name not in response_data.columns:
            raise ValueError(f'{response_name} not in the dataset')

        # clean the data for drug and target-specific entries
        logging.info('Cleaning the input datasets..')
        response_data = response_data[response_name].dropna()
        # Both are arrays now.
        common_cell_lines = sorted(list(set(exp_data.index) & set(
            response_data.index)), reverse=False)
        if not common_cell_lines:
            raise ValueError(
                f'No cell lines were common for the {response_name}. Exiting!')
        else:
            logging.info(
                f'Identified a total of {len(common_cell_lines)} samples in input data')
            self.exp_data = exp_data.loc[common_cell_lines]
            self.response_data = response_data.loc[common_cell_lines]
            self.target_var = response_name
            self.model_type = model_type
            self.response_name = response_name
            # self.hcg_list = common_genes

    def get_train_val_test_kSplits(self, n_splits=5, test_size=0.2, seed=198716):
        '''
        Performs train-val-test split and returns such dataframes for further analysis.
        The test data is outside of these splits
        '''
        num_samples = self.exp_data.shape[0]

        logging.info(f"Performing k-fold split over {num_samples} samples...")

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
        self.train_cellLines = [
            i for i in self.exp_data.index if i not in test_cellLines]
        train_pos_full = all_idx[n_test:]
        self.mean_y = np.mean(
            self.response_data.iloc[train_pos_full].values.reshape(-1, 1))
        self.mean_exp = np.mean(
            self.exp_data.loc[self.train_cellLines].values, axis=0)
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

    def train_instance(self, G_train, y_train, G_val, y_val,
                       W, lmbda, lr, hY, tuning_run=False,
                       num_epochs=1000, rand_seed=198716):
        '''
        For n epochs, and a set of hyper-params, learn the matrices.
        Strictly limit to 2 cores for OPEN_BLAS (major bottleneck if using
        all the available cores)
        '''
        train_losses_Y = []
        val_losses_Y = []
        G_train = np.asarray(G_train)
        G_val = np.asarray(G_val)
        if not tuning_run:
            G_train = (G_train - self.mean_exp) / self.std_exp
            G_val = (G_val - self.mean_exp) / self.std_exp
        hY = np.asarray(hY).reshape(-1, 1)     # shape (k, 1) or (k,)

        # use threadpool_limits in threadpoolctl
        with threadpool_limits(limits=2, user_api='blas'):

            # print(threadpool_info())
            for _ in range(num_epochs):
                Z_train = G_train @ W
                multi_matrix = np.linalg.pinv(Z_train.T @ Z_train) @ Z_train.T
                hY = multi_matrix @ y_train
                hY = np.asarray(hY).reshape(-1, 1)   # (20,1)

            # logging.info('Latent vectors dimensions:', hD.shape, hI.shape)
                train_pred_Y = Z_train @ hY
                train_err_Y = train_pred_Y - y_train

            # logging.info('Error vectors dimensions:', err_d.shape, err_i.shape)
                train_err_Y = np.asarray(train_err_Y).reshape(-1, 1)
                # train_err_i = np.asarray(train_err_i).reshape(-1, 1)
            # logging.info('Error vectors dimensions after reshaping:',
            # add each train loss to a list
                train_losses_Y.append(train_err_Y)
                grad_y = 2 * G_train.T @ (train_err_Y @ hY.T)
                grad_W = 2 * (grad_y) + lmbda * np.sign(W)
                W -= lr * grad_W
                val_loss_y = self.validate_instance(G_val=G_val, y_val=y_val,
                                                    W=W, hY=hY)
                val_losses_Y.append(val_loss_y)

        if tuning_run:
            # capture only the losses in case of tuning instance
            return train_losses_Y, val_losses_Y
        else:
            return W, hY  # the losses will be computed again

    def validate_instance(self, G_val, y_val, W, hY):
        '''
        For n epochs, and a set of hyper-params, learn the matrices
        '''

        G_val = np.asarray(G_val)       # shape (n_samples, n_genes)
        Y_val = np.array(y_val).reshape(-1, 1)
        hY = np.asarray(hY).reshape(-1, 1)
        Z = G_val @ W

        pred_Y = Z @ hY
        err_Y = pred_Y - Y_val
        # logging.info('Error vectors dimensions:', err_d.shape, err_i.shape)
        err_Y = np.asarray(err_Y).reshape(-1, 1)

        return err_Y  # return the losses

    # def hyperparam_tuner(self, min_lr=0.0001, max_lr=0.1, lr_step=0.005, num_epochs=100, rand_seed=198716,
    #                      min_lmbda=0.001, max_lmbda=0.1, lmbda_step=0.005, minK=10, maxK=50, Kstep=5):
    #     logging.info('Hyperparam configurations:')
    #     lr_vector = np.arange(start=min_lr, stop=max_lr, step=lr_step)
    #     lmbda_vector = np.arange(
    #         start=min_lmbda, stop=max_lmbda, step=lmbda_step)
    #     k_vector = np.arange(start=minK, stop=maxK, step=Kstep)
    #     # itertools.product will be consumed at first call itself, covert to a list
    #     all_combos = list(product(k_vector, lr_vector, lmbda_vector))
    #     self.num_tuning_trials = len(all_combos)
    #     logging.info(
    #         f'total number of hyper-param combinations: {self.num_tuning_trials}')
    #     print(
    #         f'total number of hyper-param combinations: {self.num_tuning_trials}')
    #     if self.num_tuning_trials > 1200:
    #         print('too many points in the grid')
    #         # break
    #     # capture for all folds and combinations
    #     losses_df = pd.DataFrame(index=range(self.num_tuning_trials))
    #     for idx, combo in enumerate(all_combos):
    #         logging.info(
    #             f'Running trial: {str(idx)} with params: K={combo[0]}, lr={combo[1]}, lmbda={combo[2]}')
    #         lr = combo[1]
    #         lmbda = combo[2]
    #         for fold, cv_split in enumerate(self.training_cv_splits):
    #             W = np.random.randn(len(self.hcg_list), combo[0])
    #             hY = np.random.randn(combo[0], 1)
    #             train_cellLines = cv_split[0]
    #             val_cellLines = cv_split[1]
    #             G_train = self.exp_data.loc[train_cellLines,
    #                                         self.hcg_list].values
    #             G_train = (G_train - self.mean_exp) / self.std_exp
    #             G_val = self.exp_data.loc[val_cellLines, self.hcg_list].values
    #             G_val = (G_val - self.mean_exp) / self.std_exp
    #             Y_train = self.response_data[train_cellLines].values.reshape(
    #                 -1, 1)
    #             Y_val = self.response_data[val_cellLines].values.reshape(-1, 1)
    #             Y_train = Y_train - self.mean_y
    #             Y_val = Y_val - self.mean_y

    #             fold_train_losses_Y,  fold_val_losses_Y = self.train_instance(G_train=G_train, y_train=Y_train,
    #                                                                           G_val=G_val,
    #                                                                           y_val=Y_val, tuning_run=True,
    #                                                                           W=W, lmbda=lmbda, lr=lr, hY=hY,
    #                                                                           num_epochs=num_epochs, rand_seed=rand_seed)

    #             losses_df.loc[idx, 'latent_dim'] = combo[0]
    #             losses_df.loc[idx, 'learning_rate'] = combo[1]
    #             losses_df.loc[idx, 'lambda'] = combo[2]
    #             losses_df.loc[idx,
    #                           f'fold{fold}_train_error'] = np.array(fold_train_losses_Y).flatten()[-1]
    #             losses_df.loc[idx,
    #                           f'fold{fold}_val_error'] = np.array(fold_val_losses_Y).flatten()[-1]

    #     return losses_df

    # @staticmethod
    # def get_best_hyperparams(loss_df: pd.DataFrame):
    #     """
    #     Retrieves the single best hyper-parameter combination by calculating the
    #     instance with the least loss.
    #     """
    #     loss_df['average_loss'] = loss_df[[
    #         f'fold{i}_val_error' for i in range(5)]].mean(axis=1)
    #     best_val_idx = loss_df['average_loss'].abs().idxmin()
    #     best_row = loss_df.iloc[best_val_idx]
    #     # logging.info('Best instance: ', str(int(best_val_idx)))
    #     best_hyperparams = {
    #         'latent_dim': int(best_row['latent_dim']),
    #         'learning_rate': best_row['learning_rate'],
    #         'lambda': best_row['lambda']
    #     }
    #     logging.info(best_hyperparams)
    #     return best_hyperparams

    def train_forge(self, min_lr=0.0001, max_lr=0.1, lr_step=0.005, rand_seed=198716,
                    min_lmbda=0.001, max_lmbda=0.1, lmbda_step=0.005, minK=10, maxK=50, Kstep=5,
                    tuning_epochs=500, training_epochs=5000, seed_val=198716):
        '''
        Main function to learn W, hD and hI matrices using specific number of epochs
        and set number of randomly initialized W.
        '''
        # check for existing losses df
        if os.path.exists(self.losses_path):
            logging.info(
                f'Losses computed previously. Loading from {self.losses_path}..')
            losses_df = pd.read_csv(self.losses_path, header=0)
        else:
            logging.info(
                f'Starting FORGE training with single random instance of W..')
            logging.info(f'Tuning hyper-params..')
            losses_df = self.hyperparam_tuner(min_lr=min_lr, max_lr=max_lr, lr_step=lr_step, num_epochs=tuning_epochs,
                                              rand_seed=seed_val,
                                              min_lmbda=min_lmbda, max_lmbda=max_lmbda, lmbda_step=lmbda_step,
                                              minK=minK, maxK=maxK, Kstep=Kstep)

        if not 'trial' in losses_df.columns:
            losses_df['trial'] = losses_df.index
        best_hyperparams = self.get_best_hyperparams(loss_df=losses_df)

        self.best_hyperparams = best_hyperparams
    # logging.info(
    #     f'Best hyper-params:\n{best_hyperparams}\n Train loss: {best_train_loss:.5f}\nValidation loss: {val_losses[best_val_idx]:.5f}')
        logging.info('Training using the best hyper-params..')
        num_latent = best_hyperparams['latent_dim']
        lr_best = best_hyperparams['learning_rate']
        best_lambda = best_hyperparams['lambda']

        G_train = self.exp_hcg.loc[self.train_cellLines].values
        G_train = (G_train - self.mean_exp) / self.std_exp
        Y_train = self.response_data[self.train_cellLines].values.reshape(
            -1, 1)

        G_test = self.exp_hcg.loc[self.test_cellLines].values
        G_test = (G_test - self.mean_exp) / self.std_exp
        Y_test = self.response_data[self.test_cellLines].values.reshape(-1, 1)
    # center the y vars with mean
        Y_train = Y_train - self.mean_y
        Y_test = Y_test - self.mean_y

        W = np.random.randn(len(self.hcg_list), num_latent)
        hY = np.random.randn(num_latent, 1)
    # get the matrices and compute losses again
        tuned_W, tuned_hY = self.train_instance(G_train=G_train,
                                                y_train=Y_train,
                                                G_val=G_test,
                                                y_val=Y_test,
                                                W=W, lmbda=best_lambda,
                                                lr=lr_best,
                                                hY=hY, tuning_run=False,
                                                num_epochs=training_epochs, rand_seed=seed_val)

        response_data = G_train @ tuned_W @ tuned_hY
        test_predictions_Y = G_test @ tuned_W @ tuned_hY
    # Compute RMSE for each output separately
        train_rmse = ((response_data - Y_train)**2).mean()**0.5
        test_rmse = ((test_predictions_Y - Y_test)**2).mean()**0.5
        logging.info(
            f"[{self.response_name}"
            f"Train RMSE ({self.model_type}): {train_rmse:.5f}, "
            f"Val RMSE ({self.model_type}): {test_rmse:.5f}"
        )

        self.W = tuned_W
        self.hY = tuned_hY
        return losses_df

    def report_metrics(self, G, W, hY, actual_y):

        pred_y = G @ W @ hY

    # Ensure consistent shape
        pred_y = pred_y.reshape(-1)

        actual_y = actual_y.reshape(-1)
    # Correlations
        pearson_corr, _ = pearsonr(actual_y, pred_y)
        spearman_corr, _ = spearmanr(actual_y, pred_y)
        # MSE
        mse = mean_squared_error(actual_y, pred_y)
        res_dict = {
            'pearson_corr': round(float(pearson_corr), 5),
            'spearman_corr': round(float(spearman_corr), 5),
            'MSE': round(float(mse), 5)
        }

        return res_dict

    # -----------------------------
    # REMOVE DataFrames before saving
    # -----------------------------

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove massive DataFrames
        state.pop("exp_data", None)
        state.pop("response_data", None)
        # state.pop("ic50_data", None)
        # logging.info(state)  # omit logging.infoing the dataframes
        return state

    # -----------------------------
    # SAVE the whole object
    # -----------------------------
    def save_forge(self, path=None):
        if path is None:
            model_name = f"{self.response_name}_{self.model_type}_gridSearch.pkl"
        else:
            model_name = path

        with open(model_name, "wb") as f:
            pickle.dump(self, f)

        logging.info(f"FORGE model saved: {model_name}")

    # -----------------------------
    # LOAD saved FORGE object
    # -----------------------------
    @staticmethod
    def load_forge(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        logging.info(f"FORGE model loaded from: {path}")
        return obj

    def run_Pipeline(self, n_splits=5, min_lr=0.0001, max_lr=0.1, lr_step=0.005, rand_seed=198716,
                     min_lmbda=0.001, max_lmbda=0.1, lmbda_step=0.005, minK=10, maxK=50, Kstep=5,
                     tuning_epochs=500, training_epochs=5000, seed_val=198716, model_path='test_forge.pkl'):
        '''
        A single method to run full pipeline from tuning and training + testin.

        Full FORGE pipeline:
      1. Split data into train/val/test
      2. Tune hyperparameters using Optuna (5-fold CV)
      3. Train final models on each fold
      4. Evaluate on held-out test set
      5. Save the full object

        '''

        losses_df_path = f"{self.response_name}_losses_data_{timestamp}.csv"
        self.losses_path = losses_df_path
        logging.info(
            "============================================================")
        logging.info(
            f"Running FORGE Pipeline for {self.response_name} pair")
        logging.info(
            f"Model path {model_path}")
        # losses_df_new.to_csv(f"{self.drug}_{self.target}_losses_data_{timestamp}.csv", index=False)
        logging.info(
            f"Losses dataset path: {losses_df_path}")
        logging.info(
            "============================================================")
        self.get_train_val_test_kSplits(
            test_size=0.2, n_splits=n_splits, seed=seed_val)
        logging.info('Detecting HCGs...')
        exp_data_train = self.exp_data.loc[self.train_cellLines]
        y_train = self.response_data.loc[self.train_cellLines]
        spearman_y = spearman_corr_fun(exp_data=exp_data_train, y=y_train)
        logging.info(
            'Spearman correlations computed..Extracting top 100 HCGs..')
        n_hcgs = 100
        top_hcgs = spearman_y.abs().sort_values(
            ascending=False).head(n_hcgs).index.tolist()
        # add the target gene to HCG if not present
        if self.model_type == 'dependency' and self.response_name not in top_hcgs:
            top_hcgs.append(self.response_name)
        print(
            f'Total number of HCGs selected: {len(top_hcgs)}. Max correlation: {max(spearman_y.abs())}, Min correlation: {min(spearman_y.abs())}')
        logging.info(f'Total number of HCGs selected: {len(top_hcgs)}')
        logging.info(
            f'Cutoff for HCG: {spearman_y.loc[top_hcgs[-1]]}')
        self.hcg_list = top_hcgs  # all of these must be in exp_data
        self.exp_hcg = self.exp_data.loc[:, self.hcg_list]
        # get mean and std on entire train data and use it to scale later
        self.mean_exp = np.array(
            np.mean(exp_data_train[self.hcg_list], axis=0))
        # check for zero-division error (never happened)
        self.std_exp = np.array(np.std(exp_data_train[self.hcg_list], axis=0))

        start_time1 = time.time()
        losses_df = self.train_forge(min_lr=min_lr,
                                     max_lr=max_lr,
                                     lr_step=lr_step,
                                     rand_seed=rand_seed,
                                     min_lmbda=min_lmbda,
                                     max_lmbda=max_lmbda,
                                     lmbda_step=lmbda_step,
                                     minK=minK,
                                     maxK=maxK,
                                     Kstep=Kstep,
                                     tuning_epochs=tuning_epochs,
                                     training_epochs=training_epochs,
                                     seed_val=seed_val
                                     )

        losses_df.to_csv(self.losses_path, index=False)
        end_time1 = time.time()
        logging.info(
            f'FORGE training completed in {end_time1 - start_time1} seconds..Running additional stats..')

        # report_metrics(G, W, hD, hI, actual_dep, actual_ic50)
        G_test = self.exp_hcg.loc[self.test_cellLines]
        G_test = ((G_test - self.mean_exp) / self.std_exp).values
        Y_test = self.response_data[self.test_cellLines].values.reshape(
            -1, 1) - self.mean_y

        test_metrics = self.report_metrics(
            G=G_test, W=self.W, hY=self.hY, actual_y=Y_test)
        logging.info(f'Stats for entire test:\n {test_metrics}')
        self.__setattr__('test_res', test_metrics)
        logging.info(
            f'Pipeline successfully completed for {self.response_name}_{self.model_type} pair. Saving FORGE model..')
        self.save_forge(path=model_path)
