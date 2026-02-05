##### Supplementary scripts in FORGE pipeline #####
##### These scripts can compute benefit scores and gene influence scores #######
##### based on the weights and latent dimensions learnt by any #######
##### FORGE class #####

import pandas as pd
import numpy as np


def computeGeneInfluence(W, h):
    '''
    Function to compute the gene influence scores based on dependency and IC50 latent representations and 
    gene weights matrix W
    Inputs:
        W - The W matrix with gene weights
        h - The latent vector with learnt representations of the outcome variable
    The code initially check for data dimensions and fails if the dimensions mismatch.
    '''
    W = np.array(W)
    h = np.array(h)
    weights = W @ h
    return weights


def computeBenefitScores(exp_mat: np.array, W: np.array, hI: np.array, hD: np.array, keep_raw: bool = False,
                         inf_df_path: str = 'gene_inf_df.csv', score_df_path: str = 'score_df.csv'):
    '''
    Function to compute the benefit scores for a set of samples. This calls the ComputeGeneInfluence function
    for generating gene influence scores, and use sample-specific expression profiles for computing the
    Benefit scores. The final Benefit scores are strictly between 0 and 1, after using a min-max scaler.
    Inputs:
        exp_mat - The gene expression matrix with genes as columns and samples as rows (should be scaled using Z-transformation)
        W - The W matrix with gene weights
        hI, hD - The latent vector with learnt representations of the outcome variables IC50 and Dependency respectively
        keep_raw - If True, add the computed raw benefit scores within the dataframe (default=False).
    Returns:
        A dataframe with samples as the row indices and scaled benefit scores as the 'benefit_score' column.
    The code initially check for data dimensions and fails if the dimensions mismatch.
    '''
    G = exp_mat.values
    dep_inf = computeGeneInfluence(W=W, h=hD)
    ic50_inf = computeGeneInfluence(W=W, h=hI)
    influence_df = pd.DataFrame(index=exp_mat.columns)
    influence_df['dep_inf'] = dep_inf
    influence_df['ic50_inf'] = ic50_inf
    influence_df['combined_inf'] = dep_inf - ic50_inf
    influence_df['scaled_combined'] = (influence_df['combined_inf'] - np.mean(
        influence_df['combined_inf'])) / np.std(influence_df['combined_inf'])
    influence_df.to_csv(inf_df_path, index=True)
    benefit_scores = G @ influence_df['scaled_combined']
    scaled_scores = (benefit_scores - np.min(benefit_scores)) / \
        (np.max(benefit_scores) - np.min(benefit_scores))
    benefit_score_df = pd.DataFrame({'raw_score': benefit_scores,
                                     'scaled_scores': scaled_scores}, index=exp_mat.index)
    benefit_score_df.to_csv(score_df_path)
    return benefit_score_df