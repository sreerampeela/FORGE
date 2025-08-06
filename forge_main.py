''' Use the env with python=3.10.18 and install all dependencies'''
"""
Joint Matrix Factorization-based method to infer therapy success using baseline gene expression.

This script implements a pipeline to stratify patients for drug response
by integrating gene expression, gene dependency, and drug sensitivity (IC50) data.
It uses a matrix factorization approach to learn latent factors that explain both
drug sensitivity and gene dependency from gene expression.

The main output is a "benefit score" for each cell line, which predicts the
utility of targeting a specific gene. The pipeline also identifies genes whose
expression patterns are most influential in this prediction.

The script can be run from the command line, for example:
python forge_main.py \
    --dependency /path/to/dependency_full_cleaned.csv \
    --ic50 /path/to/erlotinib_ic50.csv \
    --expression /path/to/Erlotinib_expression.csv \
    --base_name "Erlotinib_EGFR" \
    --target_gene "EGFR"
"""

# Standard library imports
# from scipy.stats import mannwhitneyu, spearmanr
# from scipy.cluster.hierarchy import fcluster
# from matplotlib.colors import LinearSegmentedColormap, Normalize
# from adjustText import adjust_text
# import umap
import os
import argparse
import pickle
import random
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram

# --- Constants ---
MAX_ITER = 1000
LEARNING_RATE = 1e-3
LAMBDA_REG = 0.01
RANDOM_SEED = 198716
TRAIN_SPLIT_RATIO = 0.7

# resource limits
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU on shared compute
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"


def load_and_prepare_data(dep_path, ic50_path, exp_path, target_gene):
    """
    Loads, cleans, and prepares the input datasets for the pipeline.

    This function performs the following steps:
    1. Reads dependency, IC50, and expression CSV files.
    2. Cleans the dependency data index.
    3. Finds the common cell lines across all three datasets.
    4. Subsets the data to only include these common cell lines.
    5. Splits the common cell lines into training and testing sets.
    6. Converts the dataframes into NumPy arrays for model training and testing.

    Args:
        dep_path (str): Path to the gene dependency data CSV file.
        ic50_path (str): Path to the drug sensitivity (IC50) data CSV file.
        exp_path (str): Path to the gene expression data CSV file.
        target_gene (str): The name of the primary gene of interest.

    Returns:
        dict: A dictionary containing all prepared data, including:
              - G_train, D_train, K_train_target (NumPy arrays for training)
              - G_test, D_test, K_test_target (NumPy arrays for testing)
              - train_cell_lines, test_cell_lines (lists of cell line names)
              - exp_data_full (DataFrame of expression data for all common cell lines)
              - gene_names (list of gene names from the expression data)
    """
    print("--- 1. Loading and Preparing Data ---")
    data_dict = {}
    try:
        print(f"Reading dependency data from {dep_path}...")
        deps_data_full = pd.read_csv(dep_path, index_col=0, header=0)
        print(f"Reading IC50 data from {ic50_path}...")
        ic50_data = pd.read_csv(ic50_path, index_col=0, header=0)
        print(f"Reading expression data from {exp_path}...")
        exp_data = pd.read_csv(exp_path, index_col=0, header=0)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find a data file: {e}", file=sys.stderr)
        sys.exit(1)

    # Clean dependency data index if needed
    if any('_' in s for s in deps_data_full.index):
        deps_data_full['cell_line'] = [
            i.split('_')[0] for i in deps_data_full.index]
        deps_data_full.set_index('cell_line', inplace=True)
        deps_data_full = deps_data_full[~deps_data_full.index.duplicated(
            keep='first')]

    # Find common cell lines
    common_cell_lines = sorted(
        list(set(deps_data_full.index) & set(exp_data.index) & set(ic50_data.index)))
    print(
        f"Found {len(common_cell_lines)} common cell lines across all datasets.")

    if len(common_cell_lines) == 0:
        print(
            "[ERROR] No common cell lines found. Please check your input files.", file=sys.stderr)
        sys.exit(1)

    # Subset data to common cell lines
    exp_data_subset = exp_data.loc[common_cell_lines]
    # Fix1: scale to Z-scores
    exp_data_subset = (exp_data_subset - exp_data_subset.mean()
                       ) / exp_data_subset.std()
    deps_data_target = deps_data_full.loc[common_cell_lines, target_gene]
    ic50_data_subset = ic50_data.loc[common_cell_lines]

    # Split into train/test
    random.seed(RANDOM_SEED)
    num_train_samples = int(
        round(len(common_cell_lines) * TRAIN_SPLIT_RATIO, 0))
    train_indices = sorted(random.sample(
        range(len(common_cell_lines)), num_train_samples))
    test_indices = sorted(
        [i for i in range(len(common_cell_lines)) if i not in train_indices])

    data_dict['train_cell_lines'] = [common_cell_lines[i]
                                     for i in train_indices]
    data_dict['test_cell_lines'] = [common_cell_lines[i] for i in test_indices]

    # Convert to numpy arrays
    G = exp_data_subset.astype(np.float32).to_numpy()
    D = ic50_data_subset.astype(np.float32).to_numpy()
    K_target = deps_data_target.astype(np.float32).to_numpy()

    data_dict['G_train'], data_dict['G_test'] = G[train_indices,
                                                  :], G[test_indices, :]
    data_dict['D_train'], data_dict['D_test'] = D[train_indices,
                                                  :], D[test_indices, :]
    data_dict['K_train_target'], data_dict['K_test_target'] = K_target[train_indices], K_target[test_indices]

    data_dict['exp_data_full'] = exp_data_subset
    data_dict['gene_names'] = exp_data_subset.columns.tolist()

    print("Data preparation complete.")
    return data_dict


def get_weight_matrix(G, D, K, lmbda=0.01, lr=1e-3, n_latent=10, max_iter=1000):
    """
    Performs matrix factorization to learn a latent space from gene expression.

    This function learns a weight matrix 'W' that projects the gene expression matrix 'G'
    into a latent space 'Z' (Z = G @ W). The latent space is optimized to simultaneously
    predict drug sensitivity 'D' and gene dependency 'K'.

    Args:
        G (np.ndarray): Gene expression matrix (n_samples, n_genes).
        D (np.ndarray): Drug sensitivity (IC50) vector/matrix (n_samples, 1).
        K (np.ndarray): Gene dependency vector/matrix (n_samples, 1).
        lmbda (float): Regularization parameter.
        lr (float): Learning rate for gradient descent.
        n_latent (int): The number of latent factors to learn.
        max_iter (int): The maximum number of iterations for optimization.

    Returns:
        tuple: A tuple containing:
               - W (np.ndarray): The learned weight matrix (n_genes, n_latent).
               - hd (np.ndarray): The learned projection from latent space to D (n_latent, 1).
               - hk (np.ndarray): The learned projection from latent space to K (n_latent, 1).
    """
    random.seed(RANDOM_SEED)

    # Ensure D and K are 2D arrays (n_samples, 1)
    if D.ndim == 1:
        D = D[:, np.newaxis]
    if K.ndim == 1:
        K = K[:, np.newaxis]

    # --- Dimension check ---
    try:
        if G.shape[0] != D.shape[0] or G.shape[0] != K.shape[0]:
            raise ValueError(
                f"Mismatch in sample dimensions: G={G.shape}, D={D.shape}, K={K.shape}")
    except ValueError as e:
        print(
            f"[ERROR] Dimension check failed in get_weight_matrix: {e}", file=sys.stderr)
        raise

    # Center to mean
    K_mean, D_mean = K.mean(), D.mean()
    K = K - K_mean
    D = D - D_mean

    n_genes = G.shape[1]
    W = np.random.randn(n_genes, n_latent)

    for iter_num in range(max_iter):
        try:
            Z = G @ W
            multi_matrix = np.linalg.pinv(Z.T @ Z) @ Z.T

            hd = multi_matrix @ D
            hk = multi_matrix @ K

            pred_d = Z @ hd
            pred_k = Z @ hk

            err_d = pred_d - D
            err_k = pred_k - K

            grad_d = 2 * G.T @ (err_d @ hd.T)
            grad_k = 2 * G.T @ (err_k @ hk.T)

            grad_W = 2 * (grad_d + grad_k) + lmbda * np.sign(W)
            W -= lr * grad_W
        except np.linalg.LinAlgError as e:
            print(
                f"[ERROR] Matrix operation failed at iteration {iter_num}: {e}", file=sys.stderr)
            print("This can happen if latent factors become collinear. Try a different seed or learning rate.", file=sys.stderr)
            raise

        if (iter_num + 1) % 100 == 0:
            print(
                f"  Iteration {iter_num+1}, Errors: D: {np.mean(err_d**2):.4f}, K: {np.mean(err_k**2):.4f}")

    return W, hd, hk


def calculate_benefit_scores(G, W, hd, hk):
    """
    Calculates benefit scores for cell lines based on learned models.

    The benefit score is defined as `predicted_dependency - predicted_drug_sensitivity`.
    A higher score indicates that the model predicts high dependency and high sensitivity
    (low IC50, as sensitivity is -IC50), suggesting a beneficial therapeutic intervention.

    Args:
        G (np.ndarray): Gene expression matrix for which to calculate scores (e.g., test set).
        W (np.ndarray): The learned weight matrix from `get_weight_matrix`.
        hd (np.ndarray): The learned projection for drug sensitivity.
        hk (np.ndarray): The learned projection for gene dependency.

    Returns:
        tuple: A tuple containing:
               - benefit_scores (np.ndarray): The calculated benefit scores (n_samples, 1).
               - pred_D (np.ndarray): The predicted drug sensitivity values.
               - pred_K (np.ndarray): The predicted gene dependency values.
    """
    # --- Dimension check ---
    try:
        if G.shape[1] != W.shape[0]:
            raise ValueError(
                f"Incompatible shapes for G @ W: {G.shape} @ {W.shape}")
        if hd.shape[0] != W.shape[1] or hk.shape[0] != W.shape[1]:
            raise ValueError(
                f"Incompatible shapes for Z @ hd/hk: Z is (n_samples, {W.shape[1]}), but hd={hd.shape}, hk={hk.shape}")
    except ValueError as e:
        print(
            f"[ERROR] Dimension check failed in calculate_benefit_scores: {e}", file=sys.stderr)
        raise

    Z_final = G @ W
    pred_D = Z_final @ hd
    pred_K = Z_final @ hk

    # Benefit score is high dependency (high K) and high sensitivity (low D, so high -D)
    benefit_scores = pred_K - pred_D

    # Normalize benefit scores to a z-score for easier interpretation
    benefit_scores = (benefit_scores - benefit_scores.mean(axis=0)
                      ) / benefit_scores.std(axis=0)

    return benefit_scores, pred_D, pred_K


def calculate_gene_influence(W, hd, hk, gene_names):
    """
    Calculates influence scores for each gene in the expression profile.

    These scores represent how much each gene's expression contributes to the
    predicted dependency and drug sensitivity.

    Args:
        W (np.ndarray): The learned weight matrix.
        hd (np.ndarray): The learned projection for drug sensitivity.
        hk (np.ndarray): The learned projection for gene dependency.
        gene_names (list): List of gene names corresponding to the rows of W.

    Returns:
        pd.DataFrame: A dataframe with genes indexed, containing IC50 effect,
                      dependency effect, and total influence scores.
    """
    print("Calculating gene-level influence scores...")
    ic50_effect = -W @ hd
    dependency_effect = W @ hk
    total_influence = ic50_effect + dependency_effect

    influence_df = pd.DataFrame({
        "Gene": gene_names,
        "IC50Effect": ic50_effect.ravel(),
        "DependencyEffect": dependency_effect.ravel(),
        "TotalInfluence": total_influence.ravel()
    }).set_index("Gene")
    return influence_df


def create_results_dataframe(data, target_gene, pred_data, cell_lines=None):
    """Assembles all prediction results into a single DataFrame."""
    print("Assembling final results DataFrame...")
    if cell_lines is None:
        cell_lines = data['test_cell_lines']
    df = pd.DataFrame({
        f'actual_dep_{target_gene}': data['K_test_target'].ravel(),
        'actual_IC50': data['D_test'].ravel(),
        f'pred_IC50_{target_gene}': pred_data['pred_D_target'].ravel(),
        f'pred_dep_{target_gene}': pred_data['pred_K_target'].ravel(),
        f'benefit_score_{target_gene}': pred_data['scores_target'].ravel(),
    }, index=cell_lines)

    return df


def analyze_latent_space(G_test, W_target, results_df, target_gene):
    """
    Performs clustering and UMAP on the latent space and plots the results with a robust and dynamic layout.
    """
    print(f"Analyzing and plotting latent space for {target_gene} model...")

    # 1. Generate Latent Embeddings
    Z_target = G_test @ W_target
    Z_scaled = StandardScaler().fit_transform(Z_target)
    Z_df = pd.DataFrame(Z_scaled, index=results_df.index, columns=[
                        f'L{i+1}' for i in range(Z_scaled.shape[1])])

    # 2. Get Clustering Information from a Temporary Clustermap

    temp_cg = sns.clustermap(Z_df, method='ward')
    ordered_indices = temp_cg.dendrogram_row.reordered_ind
    ordered_df = Z_df.iloc[ordered_indices]
    linkage_matrix = temp_cg.dendrogram_row.linkage
    return ordered_df, linkage_matrix


def main():
    """Main function to run the entire analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Predictive Biomarker Pipeline using Matrix Factorization.")
    parser.add_argument("--dependency", required=True,
                        help="Path to dependency data CSV.")
    parser.add_argument("--ic50", required=True, help="Path to IC50 data CSV.")
    parser.add_argument("--expression", required=True,
                        help="Path to expression data CSV.")
    parser.add_argument("--base_name", required=True,
                        help="Base name for all output files (e.g., 'EGFR_Erlotinib').")
    parser.add_argument("--target_gene", required=True,
                        help="The target gene of interest (e.g., 'EGFR').")
    parser.add_argument("--num_latent", default=20,
                        help="Number of latent features to consider (default = 20)")
    args = parser.parse_args()

    # create an output dir
    os.makedirs(args.base_name, exist_ok=True)
    output_files_prefix = f"{args.base_name}/{args.base_name}"
    # --- 1. Load Data ---
    data = load_and_prepare_data(
        args.dependency, args.ic50, args.expression, args.target_gene)

    # --- 2. Run Matrix Factorization for Target Gene ---
    print(
        f"\n--- 2. Running Matrix Factorization for Target Gene: {args.target_gene} ---")
    W_target, hd_target, hk_target = get_weight_matrix(
        G=data['G_train'], D=data['D_train'], K=data['K_train_target'],
        lmbda=LAMBDA_REG, lr=LEARNING_RATE, n_latent=int(args.num_latent), max_iter=MAX_ITER
    )

    # --- 3. Calculate Benefit Scores and Predictions ---
    print("\n--- 3. Calculating Benefit Scores on Test Set ---")
    scores_target, pred_D_target, pred_K_target = calculate_benefit_scores(
        data['G_test'], W_target, hd_target, hk_target)
    pred_data = {
        'scores_target': scores_target, 'pred_D_target': pred_D_target, 'pred_K_target': pred_K_target,
    }

    # --- 4. Assemble and Save Results ---
    print("\n--- 4. Assembling and Saving Results ---")
    results_df = create_results_dataframe(
        data, args.target_gene, pred_data)
    results_output_path = f"{output_files_prefix}_test_results.csv"
    results_df.to_csv(results_output_path)
    print(f"Test set results saved to {results_output_path}")

    # --- 5. Calculate and Save Gene Influence ---
    influence_df = calculate_gene_influence(
        W=W_target, hd=hd_target, hk=hk_target, gene_names=data['gene_names'])
    influence_output_path = f"{output_files_prefix}_gene_influence.csv"
    influence_df.to_csv(influence_output_path)
    print(f"Gene influence scores saved to {influence_output_path}")

    ordered_df, linkage_matrix = analyze_latent_space(G_test=data['G_test'],
                                                      W_target=W_target,
                                                      results_df=results_df,
                                                      target_gene=args.target_gene)

    # --- 7. Save Intermediate Objects ---
    print("\n--- 7. Saving Intermediate Objects ---")
    intermediate_objs = {
        f'W_{args.target_gene}': W_target, f'hd_{args.target_gene}': hd_target, f'hk_{args.target_gene}': hk_target,
        "train_cell_lines": data['train_cell_lines'], "test_cell_lines": data['test_cell_lines'],
        'ordered_dataset': ordered_df, 'linkage_mat': linkage_matrix
    }
    intermediate_output_path = f"{args.base_name}/{args.base_name}_intermediate_objects.pkl"
    with open(intermediate_output_path, 'wb') as f:
        pickle.dump(intermediate_objs, f)
    print(f"Intermediate model objects saved to {intermediate_output_path}")

    print("\n--- Pipeline Finished Successfully! ---")


if __name__ == "__main__":
    main()
