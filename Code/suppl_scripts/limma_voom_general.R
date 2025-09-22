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
    default = 4, help = "Number of cores for parallel processing [default %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# --- Parallel setup ---
register(MulticoreParam(opt$parallel))

# --- Load data ---
message("Loading expression matrix...")
expr <- read.csv(file = opt$expression, header = TRUE, row.names = 1)
# samples as rows
expr <- t(expr) # samples as cols
message('Transposed the expression matrix..')
expr <- as.data.frame(expr, stringsAsFactors = FALSE)
# head(expr)
meta <- data.frame(sample = colnames(expr))
rownames(meta) <- colnames(expr)
# head(meta)
# Check alignment
if (!all(colnames(expr) == rownames(meta))) {
  stop("Sample names in expression matrix and metadata do not match!")
}

# --- Dummy design (no factors, intercept only) ---
design <- model.matrix(~ 1, data = meta)

# --- DGEList + voom pipeline ---
# head(expr)
y <- DGEList(counts = expr)
keep <- filterByExpr(y, design)
y <- y[keep, , keep.lib.sizes = FALSE]
y <- calcNormFactors(y)
v <- voom(y, design, plot = FALSE)

# voom normalized expression
exp <- v$E
exp <- t(exp) # samples as rows

write.csv(exp, opt$outfile, row.names = TRUE)
message(paste("Voom-normalized expression written to:", opt$outfile))
