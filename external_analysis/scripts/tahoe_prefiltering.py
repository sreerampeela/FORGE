import os
import pandas as pd
import numpy as np
import scanpy as sc
import pyarrow.dataset as ds
import anndata as ad
from scipy.stats import median_abs_deviation

# load the metadata
obs_metadata_full = pd.read_parquet('obs_fullMetadata_tahoe100M.parquet')
print("Metadata loaded")
drug_name = "Erlotinib"
print(f"Running analysis for {drug_name}")
# filter the data
drug_metadata = obs_metadata_full[(obs_metadata_full['drug'] == drug_name) & (
    obs_metadata_full["pass_filter"] == "full")].copy()

drug_barcodes = drug_metadata['BARCODE_SUB_LIB_ID'].unique()
drug_plates = drug_metadata["plate"].unique()
del obs_metadata_full
print(
    f"Detected {len(drug_barcodes)} cells among {len(drug_plates)} plates for {drug_name}")

# custom functions
# filtering genes based on MAD deviations


def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


# code
all_adatas = []
for plate_id in drug_plates:
    # load to disk initially
    plate_adata = sc.read_h5ad(f"{plate_id}_annData.h5ad", backed='r')
    # read the drug-based barcodes
    drug_adata = plate_adata[plate_adata.obs_names.isin(
        drug_barcodes)].to_memory()
    del plate_adata  # free up memory
    # mitochondrial genes
    drug_adata.var["mito"] = drug_adata.var_names.str.startswith("MT-")
# ribosomal genes
    drug_adata.var["ribo"] = drug_adata.var_names.str.startswith(
        ("RPS", "RPL"))
# hemoglobin genes.
    drug_adata.var["hb"] = drug_adata.var_names.str.contains("^HB[^(P)]")
    gene_mask = drug_adata.var.any(axis=1)
    filtered_adata = drug_adata[:, gene_mask]
    drug_adata = drug_adata.to_memory()  # full matrix to be loaded into memory
    sc.pp.calculate_qc_metrics(
        drug_adata, qc_vars=["mito", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)
    drug_adata.obs["outlier"] = (
        is_outlier(drug_adata, "log1p_total_counts", 5)
        | is_outlier(drug_adata, "log1p_n_genes_by_counts", 5)
        | is_outlier(drug_adata, "pct_counts_in_top_20_genes", 5))
    drug_adata_filtered = drug_adata[~drug_adata.obs.outlier].copy()

    # Filter genes that are expressed in less than 200 cells
    gene_mask = drug_adata_filtered.var["log1p_total_counts"] > np.log1p(200)
    drug_adata_filtered = drug_adata_filtered[:, gene_mask]
    print(
        f"Number of cells after filtering of low quality cells: {drug_adata_filtered.n_obs}")
    # save temporary files
    drug_adata_filtered.write_h5ad(f"{plate_id}_{drug_name}_preprocessed.h5ad")
    all_adatas.append(drug_adata_filtered)


# concatenate all the adatas
merged_adata = ad.concat(all_adatas, axis=0, join='outer',
                         label='plate_id', keys=drug_plates, index_unique='-')
print("All plates merged as anndata object..")
merged_adata.write_h5ad(f"{drug_name}_preprocessed_mergedAnnData.h5ad")
