import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import pickle
import random
import os


def compute_steiner_subgraph(fullPPI, terminal_nodes, n_iter, rand_prop=0.7, label='subgraph', save_path='.', seed=42):
    """
    Compute and merge multiple Steiner subgraphs over random samples of terminal nodes.

    Args:
        fullPPI (nx.Graph): The full PPI network.
        terminal_nodes (list): Node set (e.g., up/downregulated genes).
        n_iter (int): Number of iterations.
        rand_prop (float): Proportion of terminal nodes to sample each time.
        label (str): Label for output file (e.g., 'Upregulated').
        save_path (str): Path to save the final GraphML file.
        seed (int): Random seed for reproducibility.

    Returns:
        nx.Graph: Merged Steiner subgraph.
    """
    random.seed(seed)
    merged_net = nx.Graph()
    terminal_nodes = list(set(terminal_nodes) & set(fullPPI.nodes))
    n_terminals = int(rand_prop * len(terminal_nodes))

    print(f"\n▶ Starting Steiner tree computation for: {label}")
    print(f" - Terminal candidates available: {len(terminal_nodes)}")
    print(f" - Terminals per iteration: {n_terminals}")
    print(f" - Total iterations: {n_iter}")

    for i in range(n_iter):
        sampled_nodes = random.sample(terminal_nodes, n_terminals)
        steiner = steiner_tree(fullPPI, sampled_nodes,
                               weight="weight", method="mehlhorn")
        merged_net = nx.compose(merged_net, steiner)

        if ((i + 1) % 10 == 0) or (i == n_iter - 1):
            print(f"   ✔ Completed iteration {i + 1}/{n_iter}")

    # Save graph
    output_file = os.path.join(save_path, f"{label}_EGFR_steinerTree.graphml")
    nx.write_graphml(merged_net, output_file)
    print(f"✅ Final Steiner tree for {label} saved to: {output_file}\n")


# # Load full PPI graph
# fullPPI = nx.read_graphml(
#     '')
# print('📂 Loaded the full PPI graph.')

# # Load data
# with open('/home/sreeramp/cancer_dependency_project/sreeram/matrix_factorisation/intermediate_objs_train_test.pkl', 'rb') as f:
#     intermediate_objs = pickle.load(f)

# # Get gene lists
# top_genes = intermediate_objs['top_20_genes']
# bottom_genes = intermediate_objs['low_20_genes']

# # Compute upregulated and downregulated Steiner subnetworks
# compute_steiner_subgraph(
#     fullPPI, top_genes, n_iter=50, rand_prop=0.7, label='Upregulated', save_path='.', seed=1987)

# compute_steiner_subgraph(
#     fullPPI, bottom_genes, n_iter=50, rand_prop=0.7, label='Downregulated', save_path='.', seed=1987)
