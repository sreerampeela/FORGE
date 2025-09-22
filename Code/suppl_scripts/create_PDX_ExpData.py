import os
import pandas as pd
import numpy as np
import requests
from requests.exceptions import SSLError
import time

# --------------------------------
# Helper: Robust request with retries
def robust_request(url, headers=None, params=None, retries=5, wait=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response
        except SSLError as e:
            print(f"SSL error on attempt {attempt+1}: {e}")
            time.sleep(wait * (attempt + 1))
        except requests.RequestException as e:
            print(f"Request error on attempt {attempt+1}: {e}")
            time.sleep(wait * (attempt + 1))
    print(f"Failed after {retries} attempts: {url}")
    return None

# --------------------------------
# Lookup Ensembl ID from HGNC symbol
def grch37_symbol_lookup(symbol, species="homo_sapiens"):
    server = "https://grch37.rest.ensembl.org"
    endpoint = f"/xrefs/symbol/{species}/{symbol}"
    headers = {"Content-Type": "application/json"}
    response = robust_request(server + endpoint, headers=headers)
    if response is None:
        return None
    try:
        res = response.json()
        if len(res) > 0:
            return str(res[0]["id"])
    except Exception as e:
        print(f"Failed to parse JSON for symbol {symbol}: {e}")
    return None

# --------------------------------
# Get gene name and length using Ensembl ID
def getGeneLen(ensembl_id):
    server = "https://grch37.rest.ensembl.org"
    endpoint = f"lookup/id/{ensembl_id}"
    headers = {"Content-Type": "application/json"}
    params = {"expand": True}
    url = f"{server}/{endpoint}"

    response = robust_request(url, headers=headers, params=params)
    if response is None:
        return None, None

    try:
        gene_info = response.json()
        gene_name = gene_info.get("display_name")
        start = gene_info.get("start")
        end = gene_info.get("end")
        if not all([gene_name, start, end]):
            return None, None
        return gene_name, abs(end - start)
    except Exception as e:
        print(f"Error parsing response for {ensembl_id}: {e}")
        return None, None

# --------------------------------
dataset_path = "../data/pdx_data/PDX_tumorSize_drugResponses.xlsx"
exp_data = pd.read_excel(dataset_path, sheet_name="RNAseq_fpkm", header=0, index_col=0)

if os.path.exists("pseudoRawCounts.csv"):
    exp_data_copy = pd.read_csv("pseudoRawCounts.csv", header=0, index_col=0)
else:
    exp_data_copy = exp_data.copy()

print("Dataset loaded..!")
# Process gene rows

R = 35  # Estimated from PE read length for GRCh37
i = 0

# Resume from last gene if applicable
if os.path.exists("lastGene.txt"):
    with open("lastGene.txt", "r") as tmp:
        last_gene = tmp.readline().strip()
    start_idx = exp_data.index.get_loc(last_gene)
else:
    start_idx = 0

for gene_id, fpkm in exp_data.iloc[start_idx:].iterrows():
    last_gene = gene_id  # ? Track for checkpointing

    gene_ensembl_id = grch37_symbol_lookup(symbol=gene_id)
    if gene_ensembl_id:
        gene_name, gene_len = getGeneLen(ensembl_id=gene_ensembl_id)
        if gene_name == gene_id and gene_len > 0:
            exp_data_copy.loc[gene_id] = ((fpkm * gene_len) / R).round().astype(int)
        else:
            print(f"Mismatch or invalid length for {gene_id}")
            exp_data_copy = exp_data_copy.drop(gene_id, errors='ignore')
    else:
        print(f"{gene_id} not mapped")
        exp_data_copy = exp_data_copy.drop(gene_id, errors='ignore')

    i += 1
    if (i % 500) == 0:
        print(f"Saving at iteration {i}, last gene: {gene_id}")
        exp_data_copy.to_csv("pseudoRawCounts.csv", index=True)
        with open("lastGene.txt", "w") as tmp:
            tmp.write(last_gene)

# Final save
exp_data_copy.to_csv("../results/pseudoRawCounts.csv", index=True)
print("Pipeline completed..raw counts matrix generated.")

