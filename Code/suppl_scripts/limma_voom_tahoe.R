# --- Load required libraries ---
suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
  library(BiocParallel)
  library(optparse)
})

# --- Define command-line options ---
option_list <- list(
  make_option(c("-e", "--expression"), type = "character", 
    help = "Path to raw expression matrix (CSV, genes as columns)"),
  make_option(c("-o", "--outfile"), type = "character", 
    help = "Path to output CSV file with voom-normalized results"),
  make_option(c("-p", "--parallel"), type = "integer", 
    default = 4, help = "Number of cores for parallel processing [default %default]"),
 make_option("--drug", type = "logical", 
    default = FALSE, help = "whether it is drug/dmso")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# --- Parallel setup ---
register(MulticoreParam(opt$parallel))

# --- Load data ---
message("Loading expression matrix...")
expr <- read.csv(file = opt$expression, header = TRUE, row.names = 1)
# head(expr)
expr <- expr[ , !(colnames(expr) %in% c('drug', 'plate_id'))]
# samples as rows
expr <- t(expr) # samples as cols
message('Transposed the expression matrix..')
expr <- as.data.frame(expr, stringsAsFactors = FALSE)
# head(expr)
meta <- data.frame(sample = colnames(expr))
rownames(meta) <- colnames(expr)
meta$cell_line <- sapply(strsplit(meta$sample, '_'), function(x) x[1])
# head(meta)
# Check alignment
if (!all(colnames(expr) == rownames(meta))) {
  stop("Sample names in expression matrix and metadata do not match!")
}

# --- Dummy design (no factors, intercept only) ---
if (opt$drug) {
    meta$plate_id <- sapply(strsplit(meta$sample, '_'), function(x) x[2])
    design <- model.matrix(~ 1)
     } else {
   
    meta$drug <- sapply(strsplit(meta$sample, '_'), function(x) x[2])
     design <- model.matrix(~ 1 + drug, data = meta)
    }

# --- DGEList + voom pipeline ---
# head(expr)
y <- DGEList(counts = expr)
y <- calcNormFactors(y)
v <- voom(y, design, plot = FALSE)

# voom normalized expression
exp <- v$E
exp <- t(exp) # samples as rows

write.csv(exp, opt$outfile, row.names = TRUE)
message(paste("Voom-normalized expression written to:", opt$outfile))
