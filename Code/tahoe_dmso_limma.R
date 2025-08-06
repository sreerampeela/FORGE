# --- 1. Load Libraries ---
# Core limma-voom packages
library(limma)
library(edgeR) 
# Parallel processing support
library(BiocParallel)
# Utility for data wrangling and saving
library(dplyr)
library(tibble)

# --- 2. Setup and Data Loading ---
# Set parallelisation mode (used by some limma/edgeR functions if supported)
register(MulticoreParam(12))

# Define output directory for results
output_dir <- "limma_voom_results"
dir.create(output_dir, showWarnings = FALSE)

# Load the counts and metadata
message("Loading counts and metadata...")
pb_counts <- read.csv('/home/sreeramp/cancer_dependency_project/sreeram/matrix_factorisation/latent_30_validation/tahoe100M_validation/cleaned_dmso_pseudobulk.csv',
                      header = TRUE, row.names = 1)
pb_metadata <- read.csv('/home/sreeramp/cancer_dependency_project/sreeram/matrix_factorisation/latent_30_validation/tahoe100M_validation/metadata_dmso_pseudobulk.csv',
                        header = TRUE, row.names = 1)

# --- 3. Data Preparation ---
message("Preparing data for analysis...")
# Align metadata with count matrix columns and transpose counts
pb_counts <- pb_counts[row.names(pb_metadata), ]
pb_counts <- t(pb_counts)

# Sanity check: ensure all sample names match perfectly
if(!all(colnames(pb_counts) == rownames(pb_metadata))){
  stop("CRITICAL ERROR: Sample names in counts and metadata do not match!")
}

# Set all strings as factors and create syntactically valid names
pb_metadata$score_cat <- factor(pb_metadata$score_cat, levels = c('low score', 'high score'))
pb_metadata$cell_line <- factor(pb_metadata$cell_line)
pb_metadata$plate_id <- factor(pb_metadata$plate_id)
levels(pb_metadata$score_cat) <- make.names(levels(pb_metadata$score_cat))

# --- 4. limma-voom Analysis Pipeline ---
message("Starting limma-voom pipeline...")

# Create the design matrix. Note: The blocking variable 'cell_line' is NOT included here.
# It will be handled by duplicateCorrelation.
design <- model.matrix(~ score_cat + plate_id, data = pb_metadata)
colnames(design) <- make.names(colnames(design)) # Clean up column names

# Create DGEList, filter low-expressed genes, and normalize
y <- DGEList(counts = pb_counts)
keep <- filterByExpr(y, design = design)
y <- y[keep, , keep.lib.sizes=FALSE]
y <- calcNormFactors(y) # TMM normalization is standard for limma-voom

# Run voom to transform count data to log2-CPM with precision weights
message("Running voom... (fast)")
v <- voom(y, design, plot = FALSE)

# Estimate the within-cell-line correlation
# This is the key step that accounts for the fact that samples from the same cell line are not independent.
message("Estimating intra-cell-line correlation with duplicateCorrelation...")
corfit <- duplicateCorrelation(v, design, block = pb_metadata$cell_line)
message(paste("Consensus correlation:", round(corfit$consensus.correlation, 3)))

# Fit the linear model for each gene, now incorporating the block correlation
message("Fitting linear model with lmFit...")
fit <- lmFit(v, design, block = pb_metadata$cell_line, correlation = corfit$consensus)

# Apply empirical Bayes moderation
message("Applying empirical Bayes moderation with eBayes...")
fit <- eBayes(fit)

# --- 5. Extract and Save Results ---
message("Analysis complete. Saving results...")

# 5a. Save the key limma objects for later use
v_path <- file.path(output_dir, "dmso_voom_EList_object.rds")
fit_path <- file.path(output_dir, "dmso_limma_fit_object.rds")
saveRDS(v, file = v_path)
saveRDS(fit, file = fit_path)
message(paste("Saved voom EList object to:", v_path))
message(paste("Saved limma fit object to:", fit_path))

# 5b. Get results for the main effect of score_cat
# The coefficient of interest is 'score_cat.high.score'
res_main_effect <- topTable(fit, coef = "score_cat.high.score", number = Inf, sort.by = "p")

# Convert row names (genes) to a column and save to CSV
res_main_effect_df <- as.data.frame(res_main_effect) %>%
  rownames_to_column(var = "gene")

main_effect_path <- file.path(output_dir, "dmso_limma_results_main_effect_score_cat.csv")
write.csv(res_main_effect_df, main_effect_path, row.names = FALSE)
message(paste("Saved main effect results to:", main_effect_path))

print("--- Top genes for AVERAGE score effect (from limma-voom) ---")
print(head(res_main_effect_df))

message("\nAll done!")