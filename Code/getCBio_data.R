# -------------------------------------------
# Download gene expression data from cBioPortal
# for EGFR-related studies with Erlotinib treatment
# -------------------------------------------

# Load required libraries
library(cbioportalR)
library(dplyr)

# Set cBioPortal database
set_cbioportal_db(db = "public")

# Load input data
message("Loading input data...")
erlotinib_samples <- trimws(readLines("../Data/erlotinib_sample_ids.txt", warn = FALSE))
genes_of_interest <- trimws(readLines("../Data/key_gene_list.txt", warn = FALSE))
message(paste("Loaded", length(erlotinib_samples), "Erlotinib-treated samples."))
message(paste("Loaded", length(genes_of_interest), "genes of interest."))

# Optional: list of relevant studies
cbio_studies <- c(
  "luad_tcga_pan_can_atlas_2018", "kirc_tcga_pan_can_atlas_2018",
  "lusc_tcga_pan_can_atlas_2018", "gbm_tcga_pan_can_atlas_2018",
  "meso_tcga_pan_can_atlas_2018", "lgg_tcga_pan_can_atlas_2018",
  "blca_tcga_pan_can_atlas_2018"
)

# Initialize empty containers
full_exp <- data.frame(matrix(ncol = length(genes_of_interest), nrow = 0))
colnames(full_exp) <- genes_of_interest
sample_metadata <- data.frame()
studies_with_rnaSeq <- c()

# Retrieve all available studies
message("Retrieving list of available studies...")
common_studies <- get_studies()
common_studies <- common_studies[common_studies$studyId %in% cbio_studies, ]

message(paste("Processing", nrow(common_studies), "studies..."))

# Loop through selected studies
for (study_name in common_studies$studyId) {
  message(paste("Processing study:", study_name))

  # Load data pack
  t1 <- tryCatch({
    cBioDataPack(cancer_study_id = study_name)
  }, error = function(e) {
    warning(paste("Failed to download data for study:", study_name))
    return(NULL)
  })
  
  if (is.null(t1)) next
  
  available_experiments <- names(experiments(t1))
  
  if ("mrna_seq_v2_rsem" %in% available_experiments) {
    studies_with_rnaSeq <- c(studies_with_rnaSeq, study_name)
    
    # Extract expression data
    e1 <- assays(t1)[["mrna_seq_v2_rsem"]]
    e1_counts <- as.data.frame(t(e1))
    
    # Filter for genes of interest
    common_genes <- intersect(genes_of_interest, colnames(e1_counts))
    
    if (length(common_genes) == 0) {
      message(paste("No common genes found in", study_name))
      next
    }

    # Filter Erlotinib-treated samples
    e1_samples <- rownames(e1_counts)
    e1_erlotinib_samples <- e1_samples[e1_samples %in% erlotinib_samples]
    
    if (length(e1_erlotinib_samples) == 0) {
      message(paste("No Erlotinib-treated samples found in", study_name))
      next
    }
    
    e1_counts_filtered <- e1_counts[e1_erlotinib_samples, common_genes, drop = FALSE]
    
    # Align gene columns
    missing_genes <- setdiff(genes_of_interest, colnames(e1_counts_filtered))
    e1_counts_filtered[missing_genes] <- NA
    e1_counts_filtered <- e1_counts_filtered[, genes_of_interest]

    # Update metadata
    study_metadata <- data.frame(
      sample_id = rownames(e1_counts_filtered),
      batch_id = study_name,
      stringsAsFactors = FALSE
    )
    
    # Combine expression and metadata
    full_exp <- rbind(full_exp, e1_counts_filtered)
    sample_metadata <- rbind(sample_metadata, study_metadata)

    message(paste("Added", nrow(e1_counts_filtered), "samples from", study_name))
    
  } else {
    message(paste("No RNA-seq data found in", study_name))
  }
}

# Final report
message(paste("Total studies with RNA-seq data:", length(unique(sample_metadata$batch_id))))
message(paste("Total samples processed:", nrow(full_exp)))

# Save results
write.csv(full_exp, '../Data/CbioExp_data.csv', row.names = TRUE)
write.csv(sample_metadata, '../Data/Cbio_mappedStudies.csv', row.names = FALSE)

message("✅ Expression data saved to '../Data/CbioExp_data.csv'")
message("✅ Metadata saved to '../Data/Cbio_mappedStudies.csv'")
