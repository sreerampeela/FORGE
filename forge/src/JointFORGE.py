import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, rankdata
from threadpoolctl import threadpool_limits, threadpool_info
import random
import pickle
import optuna
import time
import logging

log_file = "forge_fullPipeline.log"


logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"   # overwrite log file on each execution
)


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

    def __init__(self, exp_path, dependency_path, ic50_path, drug_name, target_name, log_file, overwrite):
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
                    "*            New FORGE Run Initialised             *\n"
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
                    "*        Continuing Previous FORGE Run             *\n"
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
                "*            New FORGE Run Initialised             *\n"
                "****************************************************"
            )
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
        logging.info('Cleaning the input datasets..')
        ic50_data = ic50_data[drug_name].dropna()
        dep_data = dep_data[target_name].dropna()
        # Both are arrays now.
        common_cell_lines = sorted(list(set(exp_data.index) & set(
            dep_data.index) & set(ic50_data.index)), reverse=False)
        if not common_cell_lines:
            raise ValueError(
                f'No cell lines were common for the {drug_name}-{target_name} pair. Exiting!')
            sys.exit(1)
        else:
            logging.info(
                f'Identified a total of {len(common_cell_lines)} samples in input data')
            self.exp_data = exp_data.loc[common_cell_lines]
            self.dep_data = dep_data.loc[common_cell_lines]
            self.ic50_data = ic50_data.loc[common_cell_lines]
            self.drug = drug_name
            self.target = target_name
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
        self.mean_dep = np.mean(
            self.dep_data.iloc[train_pos_full].values.reshape(-1, 1))
        self.mean_ic50 = np.mean(
            self.ic50_data.iloc[train_pos_full].values.reshape(-1, 1))
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
        with threadpool_limits(limits=2, user_api='blas'):
            for epoch in range(num_epochs):
                Z_train = G_train @ W
                multi_matrix = np.linalg.pinv(Z_train.T @ Z_train) @ Z_train.T
                hD = multi_matrix @ D_train
                hI = multi_matrix @ I_train
                hD = np.asarray(hD).reshape(-1, 1)   # (20,1)
                hI = np.asarray(hI).reshape(-1, 1)

            # logging.info('Latent vectors dimensions:', hD.shape, hI.shape)
                train_pred_d = Z_train @ hD
                train_pred_i = Z_train @ hI

                train_err_d = train_pred_d - D_train
                train_err_i = train_pred_i - I_train
            # logging.info('Error vectors dimensions:', err_d.shape, err_i.shape)
                train_err_d = np.asarray(train_err_d).reshape(-1, 1)
                train_err_i = np.asarray(train_err_i).reshape(-1, 1)
            # logging.info('Error vectors dimensions after reshaping:',
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
            # if (epoch + 1) % 100 == 0:

            #     logging.info(
            #         f'Epoch {epoch + 1}: Training loss:\n  Dependency: {np.mean(train_err_d):.5f}\n   IC50: {np.mean(train_err_i):.5f}')
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
        # logging.info('Error vectors dimensions:', err_d.shape, err_i.shape)
        err_d = np.asarray(err_d).reshape(-1, 1)
        err_i = np.asarray(err_i).reshape(-1, 1)
        # logging.info('Error vectors dimensions after reshaping:',
        #   err_d.shape, err_i.shape)
        val_loss = (np.mean(err_d**2) + np.mean(err_i ** 2)
                    ) * 0.5  # average of both MSEs

        # val_losses.append(round(val_loss, 5))
        return val_loss

    def optuna_tuner(self, seed_val, num_epochs=1000, n_trials=5):
        """
        Hyperparameter tuning using Optuna based on validation loss. For each hyper-param all the
        five folds are tested, and hyper-param with least error will be returned.
        """
        sampler = optuna.samplers.TPESampler(seed=198716) # use the same set of random trials
        logging.info(
            f"\n--- Starting Optuna search for {self.drug} - {self.target} ---")

    # --------------------------
    # Objective function
    # --------------------------
        def objective(trial):
            np.random.seed(seed_val)

            # initialise losses
            fold_train_losses, fold_val_losses = [], []

        # Hyperparameters
            K = trial.suggest_categorical("latent_dim", list(range(10, 75, 5)))
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            lmbda = trial.suggest_float("lambda", 1e-5, 1e-1, log=True)
            W_dim1 = len(self.hcg_list)

        # --------------------------
        # Training loop for all folds
        # --------------------------
            with threadpool_limits(limits=2, user_api='blas'):
                for idx, fold_idx in enumerate(self.training_cv_splits):
                    # Initialize W, hD, hI separately for each fold
                    W = np.random.randn(W_dim1, K)  # gxk matrix
                    hD = np.random.randn(K, 1)
                    hI = np.random.randn(K, 1)
                    train_cellLines = fold_idx[0]
                    val_cellLines = fold_idx[1]
                    G_train = self.exp_data.loc[train_cellLines,
                                                self.hcg_list]
                    G_train = ((G_train - self.mean_exp) / self.std_exp).values
                    G_val = self.exp_data.loc[val_cellLines,
                                              self.hcg_list]
                    G_val = ((G_val - self.mean_exp) / self.std_exp).values
                    D_train = self.dep_data[train_cellLines].values.reshape(
                        -1, 1)
                    D_val = self.dep_data[val_cellLines].values.reshape(-1, 1)
                    D_train = D_train - self.mean_dep
                    D_val = D_val - self.mean_dep
                    I_train = self.ic50_data[train_cellLines].values.reshape(
                        -1, 1)
                    I_val = self.ic50_data[val_cellLines].values.reshape(-1, 1)
                    I_train = I_train - self.mean_ic50
                    I_val = I_val - self.mean_ic50
                    tuned_W, tuned_hD, tuned_hI, train_losses, val_losses = self.train_instance(G_train=G_train, D_train=D_train,
                                                                                                I_train=I_train, G_val=G_val,
                                                                                                D_val=D_val, I_val=I_val,
                                                                                                W=W, lmbda=lmbda, lr=lr, hD=hD, hI=hI,
                                                                                                num_epochs=num_epochs, rand_seed=seed_val)

                # we are not storing intermediate matrices as we want to retrain and get them later
                    fold_train_losses.append(train_losses[-1])
                    fold_val_losses.append(val_losses[-1])

            # aggregate train and val losses for all folds
                mean_train_loss = np.mean(fold_train_losses)
                mean_val_loss = np.mean(fold_val_losses)
                # print(
                #     f'end of {trial.number}. Average loss across all folds: Training: {mean_train_loss:.3f}, validation: {mean_val_loss:.3f}')
            # # store the mean train and val losses per trial
            #     self.__setattr__(
            #         f'trial{trial.number}_train_loss', mean_train_loss)
            #     self.__setattr__(
            #         f'trial{trial.number}_val_loss', mean_val_loss)

                return mean_val_loss

    # --------------------------
    # Create study
    # --------------------------
        instance_name = f'{self.drug}_{self.target}_optuna'
        # clean prevous optuna db
        if os.path.exists(f'{instance_name}.db'):
            os.system(f'rm -rf {instance_name}.db')
        study = optuna.create_study(
            study_name=instance_name,
            direction="minimize",
            storage=f"sqlite:///{instance_name}.db",
            load_if_exists=True
        )
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials)  # for testing
        end_time = time.time()
        logging.info(
            f'Hyper-param tuning completed in {end_time - start_time} seconds..')
        # logging.info("\nBest params:", study.best_params)

        return study.best_params

    def train_forge(self, optuna_trials=100, quiet=False,
                    tuning_epochs=500, training_epochs=5000, seed_val=198716):
        '''
        Main function to learn W, hD and hI matrices using specific number of epochs
        and set number of randomly initialized Ws. The final matrices are aggregated using
        mean of each bootstrap instance.
        '''
        logging.info(
            f'Starting FORGE training with single random instance of W..')
        logging.info(f'Tuning hyper-params..')
        # if instance_name is None:
        #     instance_name = f'{self.drug}_{self.target}_optuna_fold{fold_name}'
        if quiet:
            optuna.logging.set_verbosity(logging.ERROR)
        # optuna_tuner(self, G_train, D_train, I_train, G_val, D_val, I_val, seed_val, num_epochs=1000, n_trials=5)
        # best_hyperparams = self.optuna_tuner(G_train=G_train, D_train=D_train, I_train=I_train,
        #                                      G_val=G_val, D_val=D_val, I_val=I_val, n_trials=optuna_trials,
        #                                      seed_val=seed_val, num_epochs=tuning_epochs, instance_name=instance_name)
        best_hyperparams = self.optuna_tuner(n_trials=optuna_trials,
                                             seed_val=seed_val,
                                             num_epochs=tuning_epochs)
        logging.info(
            f'Best hyper-params:\n{best_hyperparams}')
        logging.info('Training entire dataset using the best hyper-params..')
        num_latent = best_hyperparams['latent_dim']
        lr_best = best_hyperparams['learning_rate']
        best_lambda = best_hyperparams['lambda']
        # n_genes = len(self.hcg_list)
        # Train using the reshaped matrices and best hyper-params
        train_cellLines = self.train_cellLines
        val_cellLines = self.test_cellLines
        G_train = self.exp_hcg.loc[train_cellLines,
                                   self.hcg_list]
        G_train = ((G_train - self.mean_exp) / self.std_exp).values
                     
        G_val = self.exp_hcg.loc[val_cellLines, self.hcg_list]
        G_val = ((G_val - self.mean_exp) / self.std_exp).values
        D_train = self.dep_data[train_cellLines].values.reshape(-1, 1)
        D_val = self.dep_data[val_cellLines].values.reshape(-1, 1)
        D_train = D_train - self.mean_dep
        D_val = D_val - self.mean_dep
        I_train = self.ic50_data[train_cellLines].values.reshape(-1, 1)
        I_val = self.ic50_data[val_cellLines].values.reshape(-1, 1)
        I_train = I_train - self.mean_ic50
        I_val = I_val - self.mean_ic50
        W = np.random.randn(G_train.shape[1],
                            num_latent)  # gxk matrix
        hD = np.random.randn(num_latent, 1)
        hI = np.random.randn(num_latent, 1)
        tuned_W, tuned_hD, tuned_hI, train_losses, val_losses = self.train_instance(G_train=G_train,
                                                                                    D_train=D_train,
                                                                                    I_train=I_train, G_val=G_val,
                                                                                    D_val=D_val, I_val=I_val,
                                                                                    W=W, lmbda=best_lambda,
                                                                                    lr=lr_best,
                                                                                    hD=hD, hI=hI,
                                                                                    num_epochs=training_epochs, rand_seed=seed_val)

        train_predictions_D = G_train @ tuned_W @ tuned_hD
        train_predictions_I = G_train @ tuned_W @ tuned_hI
        tuned_train_loss = 0.5 * \
            (((train_predictions_D - D_train) ** 2).mean() +
             ((train_predictions_I - I_train) ** 2).mean())
        # average losses
        val_predictions_D = G_val @ tuned_W @ tuned_hD
        val_predictions_I = G_val @ tuned_W @ tuned_hI
        tuned_val_loss = 0.5 * (((val_predictions_D - D_val) ** 2).mean() +
                                ((val_predictions_I - I_val) ** 2).mean())
        logging.info(
            f"Final training loss: {tuned_train_loss:.5f}. "
            f"Test loss: {tuned_val_loss:.5f}"
        )
        self.W = tuned_W
        self.hD = tuned_hD
        self.hI = tuned_hI

        # setattr(self, f"fold{str(idx)}_train_predictions_dep",
        #         train_predictions_D)
        # setattr(self, f"fold{str(idx)}_train_predictions_IC50",
        #         train_predictions_I)

        # setattr(self, f"fold{str(idx)}_val_predictions_dep",
        #         val_predictions_D)
        # setattr(
        #     self, f"fold{str(idx)}_val_predictions_IC50", val_predictions_I)

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
        # logging.info(state)  # omit logging.infoing the dataframes
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

    def run_Pipeline(self, n_splits, seed_val, tuning_epochs=500, quiet=False,
                     training_epochs=5000, optuna_trials=100, model_path='test_forge.pkl'):
        '''
        A single method to run full pipeline from tuning and training + testin.

        Full FORGE pipeline:
      1. Split data into train/val/test
      2. Tune hyperparameters using Optuna (5-fold CV)
      3. Train final models on each fold
      4. Evaluate on held-out test set
      5. Save the full object

        '''

        logging.info(
            "============================================================")
        logging.info(
            f"Running FORGE Pipeline for {self.drug}-{self.target} pair")
        logging.info(
            "============================================================")

        self.get_train_val_test_kSplits(
            test_size=0.2, n_splits=n_splits, seed=seed_val)

        logging.info('Detecting HCGs...')
        exp_data_train = self.exp_data.loc[self.train_cellLines]
        dep_train = self.dep_data.loc[self.train_cellLines]
        ic50_train = self.ic50_data.loc[self.train_cellLines]
        spearman_dep = spearman_corr_fun(exp_data=exp_data_train, y=dep_train)
        spearman_ic50 = spearman_corr_fun(
            exp_data=exp_data_train, y=ic50_train)
        logging.info(
            'Spearman correlations computed..Extracting top 100 HCGs..')
        n_hcgs = 100
        top_hcgs_dep = spearman_dep.abs().sort_values(
            ascending=False).head(n_hcgs).index.tolist()
        top_hcgs_ic50 = spearman_ic50.abs().sort_values(
            ascending=False).head(n_hcgs).index.tolist()
        total_hcgs = list(set(top_hcgs_dep).union(set(top_hcgs_ic50)))
        # add the target if not in HCG list
        if self.target not in total_hcgs:
            total_hcgs.append(self.target)
        # total_hcgs = list(set(total_hcgs))  # maybe a duplicate is added here
        print('Inferred number of HCGs:', len(total_hcgs))
        logging.info(f'Total number of HCGs selected: {len(total_hcgs)}')
        logging.info(
            f'Cutoffs for IC50: {spearman_ic50.loc[top_hcgs_ic50[-1]]} and dependency: {spearman_dep.loc[top_hcgs_dep[-1]]}')

        self.hcg_list = total_hcgs  # all of these must be in exp_data
        self.exp_hcg = self.exp_data.loc[:, self.hcg_list]
        self.mean_exp = self.exp_hcg.mean(axis = 0)
        self.std_exp = self.exp_hcg.std(axis = 0)
        start_time1 = time.time()
        self.train_forge(optuna_trials=optuna_trials,
                         tuning_epochs=tuning_epochs,
                         seed_val=seed_val, training_epochs=training_epochs, quiet=quiet)

        end_time1 = time.time()
        logging.info(
            f'FORGE training completed in {end_time1 - start_time1} seconds..Running test..')
        test_ids = self.test_cellLines
        G_test = self.exp_hcg.loc[test_ids]
        G_test = ((G_test - self.mean_exp) / self.std_exp).values
        # 1D vector
        D_test = self.dep_data.loc[test_ids].values.reshape(-1, 1)
        # 1D vector
        I_test = self.ic50_data.loc[test_ids].values.reshape(-1, 1)
        # scaling the dependency and IC50 values
        D_test = D_test - self.mean_dep
        I_test = I_test - self.mean_ic50

        # # run test on all 5-folds Ws
        # for i in range(5):
        #     fold_hD = getattr(self, f"fold{str(i)}_hD")
        #     fold_W = getattr(self, f"fold{str(i)}_W")
        #     fold_hI = getattr(self, f"fold{str(i)}_hI")
        #     test_predictions_D = G_test @ fold_W @ fold_hD
        #     test_predictions_I = G_test @ fold_W @ fold_hI
        #     tuned_test_loss = 0.5 * \
        #         (((test_predictions_D - D_test) ** 2).mean() +
        #          ((test_predictions_I - I_test) ** 2).mean())
        #     logging.info(
        #         f"Final test loss for fold {i}: {tuned_test_loss:.5f}. "
        #     )
        # report_metrics(G, W, hD, hI, actual_dep, actual_ic50)
        test_metrics = self.report_metrics(G=G_test, W=self.W, hD=self.hD,
                                           hI=self.hI, actual_dep=D_test, actual_ic50=I_test)
        logging.info(test_metrics)
        # self.__setattr__(f'Fold{str(i)}_test_res', test_metrics)
        logging.info(
            f'Pipeline successfully completed for {self.drug}-{self.target} pair. Saving FORGE model..')
        self.save_forge(path=model_path)
