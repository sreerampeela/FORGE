import io
import os
import scanpy as sc
import pyarrow.dataset as ds
import gcsfs
import pandas as pd
import tqdm
import numpy as np

# Initialize GCS file system
fs = gcsfs.GCSFileSystem()
for i in range(7, 10, 1):
      plate_id = f"plate{i}"
    file_path = f"../results/{plate_id}_annData.h5ad" #
    test_plate = f"arc-ctc-tahoe100/2025-02-25/h5ad/{plate_id}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad" #
    try: #
        print("Downloading from link:", test_plate) #
        fs.get(rpath=test_plate, lpath=file_path) #
        print(f"Plate {i} data downloaded..") #
    except Exception as e: #
        print(f"? Failed: {e}")

# get DMSO pseudobulk from plate9
plate_id = 'plate9'
adata = sc.read_h5ad(f'{plate_id}_annData.h5ad', backed='r')
adata_DMSO = adata[adata.obs["drug"] == "DMSO_TF"]
adata_dense = adata_DMSO.to_memory()
X = adata_dense.X.toarray()
cell_names = adata_dense.obs["cell_name"].values
gene_names = adata_dense.var_names

# Map to store aggregated counts
pseudobulk_dict = {}

# Build list of unique cell lines
unique_cells = np.unique(cell_names)

# Loop with progress bar and aggregate
for cell in tqdm.tqdm(unique_cells, desc="Aggregating pseudobulk"):
    idx = np.where(cell_names == cell)[0]
    pseudobulk_expr = X[idx].sum(axis=0)
    pseudobulk_dict[cell] = pseudobulk_expr

# Convert to DataFrame
dmso_pb = pd.DataFrame.from_dict(
    pseudobulk_dict, orient="index", columns=gene_names)
dmso_pb['drug'] = 'DMSO_TF'

# get adata for erlotinib
adata_erlotinib = adata[adata.obs["drug"] == "erlotinib"]
adata_dense = adata_erlotinib.to_memory()
X = adata_dense.X.toarray()
cell_names = adata_dense.obs["cell_name"].values
gene_names = adata_dense.var_names

# Map to store aggregated counts
pseudobulk_dict = {}

# Build list of unique cell lines
unique_cells = np.unique(cell_names)

# Loop with progress bar and aggregate
for cell in tqdm.tqdm(unique_cells, desc="Aggregating pseudobulk"):
    idx = np.where(cell_names == cell)[0]
    pseudobulk_expr = X[idx].sum(axis=0)
    pseudobulk_dict[cell] = pseudobulk_expr

# Convert to DataFrame
drug_pb = pd.DataFrame.from_dict(
    pseudobulk_dict, orient="index", columns=gene_names)
drug_pb['drug'] = 'erlotinib'

# merge both dataframes
merged_plate9_pb = pd.concat([dmso_pb, drug_pb], axis=0)
merged_plate9_pb.to_csv('../data/tahoe_data/erlotinib_dmso_pb.csv', index=True)

# create pseudobulk for all plates erlotinib
drug_pbs = []
for i in range(7, 10, 1):
    plate_id = f"plate{i}"
    file_path = f"../results/{plate_id}_annData.h5ad"
    adata = sc.read_h5ad(file_path, backed='r')
    adata_erlotinib = adata[adata.obs["drug"] == "erlotinib"]
    adata_dense = adata_erlotinib.to_memory()
    X = adata_dense.X.toarray()
    cell_names = adata_dense.obs["cell_name"].values
    gene_names = adata_dense.var_names

    # Map to store aggregated counts
    pseudobulk_dict = {}

    # Build list of unique cell lines
    unique_cells = np.unique(cell_names)

    # Loop with progress bar and aggregate
    for cell in tqdm.tqdm(unique_cells, desc=f"Aggregating pseudobulk {plate_id}"):
        idx = np.where(cell_names == cell)[0]
        pseudobulk_expr = X[idx].sum(axis=0)
        pseudobulk_dict[cell] = pseudobulk_expr

    # Convert to DataFrame
    drug_pb = pd.DataFrame.from_dict(
        pseudobulk_dict, orient="index", columns=gene_names)
    drug_pb['plate_id'] = plate_id
    drug_pb['pb_id'] = drug_pb['plate_id'].astype(str) + '_' + drug_pb.index.astype(str)
    drug_pb.set_index('pb_id', inplace=True)
    drug_pbs.append(drug_pb)

merged_drug_pb = pd.concat(drug_pbs, axis=0)
merged_drug_pb.to_csv('../data/tahoe_data/drug_pseudobulk_raw.csv', index=True)
