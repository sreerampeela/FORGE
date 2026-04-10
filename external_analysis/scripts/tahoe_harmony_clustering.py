import os
import pandas as pd
import numpy as np
import scanpy as sc
import pyarrow.dataset as ds
import anndata as ad
from scipy.stats import median_abs_deviation
from scipy.sparse import issparse
import time
import bbknn  # alternative to Harmony

drug_name = "Erlotinib"

# # load the data and perform some QC
merged_adata = sc.read_h5ad(f"{drug_name}_preprocessed_mergedAnnData.h5ad")
merged_adata.layers["raw_counts"] = merged_adata.X.copy()
print("loaded annData object..running doublet detection..")

# # Doublet detection with scrublet (note: scrublet returns a **copy**, so capture it!)
# # Detect top 10% HVGs first (faster and more robust)
start_time = time.time()
sc.pp.highly_variable_genes(merged_adata, n_top_genes=int(
    merged_adata.shape[1]*0.1), flavor='seurat_v3')
end_time = time.time()
# print(f"Time taken to detect HVGs: {end_time - start_time:.2f} seconds")
print("HVGs detected for SCRUBLET..")
print(f"Time taken to detect HVGs: {end_time - start_time:.2f} seconds")
adata_hvg = merged_adata[:, merged_adata.var.highly_variable].copy()

# # Scrublet with fixed parameters (based on https://doi.org/10.1016/j.ccell.2023.12.012)
# # scrublet with no fixed threshold
print("Running SCRUBLET..")
start_time = time.time()
merged_adata_scrub = sc.pp.scrublet(
    adata_hvg,  n_prin_comps=50, batch_key='plate_id',
    expected_doublet_rate=0.1, log_transform=False, verbose=True,
    threshold=None, n_neighbors=50, copy=True)
print("SCRUBLET done..")
end_time = time.time()
print(f"Time taken to run SCRUBLET: {end_time - start_time:.2f} seconds")
# #  Keep only **non-doublets** (where predicted_doublet is False)
merged_adata_nodoublets = merged_adata[~merged_adata_scrub.obs['predicted_doublet']].copy(
)
num_doublets = merged_adata_scrub.obs['predicted_doublet'].sum()
pct_doublets = (num_doublets * 100) / merged_adata_scrub.n_obs
# merged_adata_nodoublets_new.write_h5ad("Erlotinib_mergedAnnData_scrublet_auto.h5ad")
print(f"Cells detected as doublets: {num_doublets} "
      f"({pct_doublets:.2f}%)")  # 36 cells detected as doublets

# save the doublets layer
merged_adata_nodoublets.write_h5ad(
    f"{drug_name}_mergedAnnData_scrublet_auto.h5ad")
del merged_adata_nodoublets

# reload the data
merged_adata_nodoublets = sc.read_h5ad(
    f"{drug_name}_mergedAnnData_scrublet_auto.h5ad")

# filter genes and cells (some hard thresholds)
sc.pp.filter_cells(merged_adata_nodoublets, min_genes=500)
sc.pp.filter_genes(merged_adata_nodoublets, min_cells=200)
# 1. Normalize counts per cell to equal depth
sc.pp.normalize_total(merged_adata_nodoublets, target_sum=1e4)

# 2. Log-transform the data (natural log)
sc.pp.log1p(merged_adata_nodoublets)

# identify HVGs (top 10% of total genes)
start_time = time.time()
sc.pp.highly_variable_genes(merged_adata_nodoublets, n_top_genes=int(
    merged_adata_nodoublets.shape[1]*0.1), flavor='cell_ranger')
end_time = time.time()
# print(f"Time taken to detect HVGs: {end_time - start_time:.2f} seconds")
print("HVGs detected for batch correction..")
print(f"Time taken to detect HVGs: {end_time - start_time:.2f} seconds")

# scale and PCA
sc.pp.scale(merged_adata_nodoublets, max_value=10)
# PCA
# using high value as more than 2.5lakh cells
sc.tl.pca(merged_adata_nodoublets, svd_solver='arpack', n_comps=50)
# Batch correction using ComBat
start_time = time.time()
print("Running Harmony for batch effect correction...")
sc.external.pp.harmony_integrate(
    merged_adata_nodoublets, key="plate_id", max_iter_harmony=20)
print("Batch correction done")
end_time = time.time()
# 159.44 seconds and 6 iters
print(f"Time taken to run Harmony: {end_time - start_time:.2f} seconds")
merged_adata_nodoublets.write_h5ad(f"{drug_name}_mergedAnnData_harmony.h5ad")
del merged_adata_nodoublets

# # Neighbors, UMAP, Clustering
merged_adata_harmony = sc.read_h5ad(f"{drug_name}_mergedAnnData_harmony.h5ad")
print("loaded Harmony object..")
# # larger n_pcs is ok for large data
sc.pp.neighbors(merged_adata_harmony, use_rep='X_pca_harmony',
                n_pcs=50, n_neighbors=20)
print("Neighborhood network computed (using Harmony PCA embeddings)..")
sc.tl.umap(merged_adata_harmony)
print("UMAP done..")

# # try different resolutions
resolutions = [0.2, 0.5, 0.75, 1.0]
for res in resolutions:
    print(f"Running Leiden clustering with resolution {res}")
    key_added = f'leiden_r{res:.1f}'  # Example: leiden_r0.4
    sc.tl.leiden(
        merged_adata_harmony,
        resolution=res,
        key_added=key_added,
        flavor="igraph"  # for large data
    )
    print(f"clustering done for resolution {res}")

# save final obj
merged_adata_harmony.write_h5ad(f"{drug_name}_mergedAnnData_clustered.h5ad")


