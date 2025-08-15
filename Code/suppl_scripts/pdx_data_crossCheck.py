import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
n_genes = 50000
np.random.seed(42)

# Simulate gene lengths (typical: 500bp to 100kb, skewed toward smaller lengths)
gene_lengths = np.random.lognormal(mean=np.log(3000), sigma=1, size=n_genes).astype(int)

# Simulate FPKM values (log-normal distribution, skewed toward low expression)
fpkm_values = np.random.lognormal(mean=0, sigma=1.5, size=n_genes)

# Read lengths to test (paired-end so fragment length ~ 2 * read length)
read_lengths = np.arange(75, 101, 5)  # 75, 80, 85, 90, 95, 100 bp
depth_values = np.arange(20, 51, 10)  # 20M, 30M, 40M, 50M mapped fragments

# Store results
results = []

for read_len in read_lengths:
    frag_len = 2 * read_len
    subtract_bp = frag_len - 1  # Effective length = L_i - subtract_bp
    for depth in depth_values:
        effective_lengths = np.maximum(gene_lengths - subtract_bp, 1)
        pseudo_counts = (fpkm_values * effective_lengths) / depth
        results.append(pd.DataFrame({
            "Read_Length": read_len,
            "Depth_M": depth,
            "Pseudo_Raw_Count": pseudo_counts
        }))

# Combine
df_results = pd.concat(results, ignore_index=True)

# Plot distribution (log scale)
plt.figure(figsize=(12, 6))
for depth in depth_values:
    subset = df_results[df_results["Depth_M"] == depth]
    plt.hist(np.log10(subset["Pseudo_Raw_Count"] + 1), bins=50, alpha=0.5, label=f"Depth {depth}M")
plt.xlabel("log10(Pseudo-Raw Count + 1)")
plt.ylabel("Number of Genes")
plt.title("Distribution of Estimated Pseudo-Raw Counts for Different Depths (All Read Lengths Pooled)")
plt.legend()
plt.show()
