## Testing FORGE implementations

This section describes how to test the different FORGE model architectures using the example dataset provided in the repository.

#### 📁 Test Dataset Overview

The test dataset includes:

Gene expression profiles for ~16,000 genes

Drug response (IC50) values for three compounds:

1. ERLOTINIB

2. AFATINIB

3. DAPORINAD

Gene dependency scores for:

1. EGFR

2. NAMPT

3. A1CF

⚠️ Note: The dataset is intentionally reduced in sample size for demonstration purposes. Model performance metrics may therefore be sub-optimal and should not be interpreted as final benchmark results.

#### 📦 Directory Setup

Before training any model, create a directory to store trained models and checkpoints:

mkdir Models


All trained model weights and intermediate checkpoints should be saved to this directory.

#### 🚀 Running FORGE Models

In the current directory, run each script as 'python3 <script>.py'. The standard options for output prefixes etc can be modified by editing the respective scripts.
