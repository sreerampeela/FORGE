#!/usr/bin/env Rscript

# --- Load required libraries ---
suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
  library(BiocParallel)
  library(tibble)
  library(dplyr)
  library(optparse)
})

# --- Define command-line options ---
option_list <- list(
  make_option(c("-e", "--expression"), type = "character", 
    help = "Path to raw expression matrix (CSV, genes as columns)"),
  make_option(c("-m", "--metadata"), type = "character", 
    help = "Path to metadata file (CSV, rows = samples)"),
  make_option(c("-f", "--factors"), type = "character", 
    help = "Comma-separated column names in metadata to use as factors (e.g., score_cat,cell_line,plate_id)"),
  make_option(c("-c", "--contrast"), type = "character", 
    help = "The main contrast of interest (e.g., score_cat.high.score)"),
  make_option(c("-o", "--outfile"), type = "character", 
    help = "Path to output CSV file with limma results"),
  make_option(c("-p", "--parallel"), type = "integer", 
    default = 4, help = "Number of cores for parallel processing [default %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# --- Parallel setup ---
register(MulticoreParam(opt$parallel))

# --- Load data ---
message("Loading expression matrix and metadata...")
expr <- read.csv(opt$expression, header = TRUE, row.names = 1)
meta <- read.csv(opt$metadata, header = TRUE, row.names = 1)

# Align and transpose
expr <- expr[row.names(meta), ]
expr <- t(expr)

# Check alignment
if (!all(colnames(expr) == rownames(meta))) {
  stop("Sample names in expression matrix and metadata do not match!")
}

# Process factor columns
factor_cols <- unlist(strsplit(opt$factors, ","))

# Design matrix
design_formula <- as.formula(paste("~", paste(factor_cols, collapse = " + ")))
design <- model.matrix(design_formula, data = meta)
colnames(design) <- make.names(colnames(design))

# DGEList + voom pipeline
y <- DGEList(counts = expr)
keep <- filterByExpr(y, design)
y <- y[keep, , keep.lib.sizes = FALSE]
y <- calcNormFactors(y)
v <- voom(y, design, plot = FALSE)
exp <- v$E # genes as rows
exp <- t(exp) # samples as rows

write.csv(exp, opt$outfile, row.names = TRUE)
message(paste("Limma-voom results written to:", opt$outfile))
