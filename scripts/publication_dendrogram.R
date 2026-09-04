#!/usr/bin/env Rscript

# Deterministic visualization only: the posterior mean C is a frozen Phase 8
# artifact, and the clustering definition is the audited Franks construction.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
  stop("Expected: latent.csv labels.csv output.png output.pdf league merges.csv")
}

latent <- read.csv(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
labels <- read.csv(args[[2]], stringsAsFactors = FALSE)
metric_names <- latent[[1]]
C <- as.matrix(latent[, -1, drop = FALSE])
storage.mode(C) <- "double"
if (!identical(metric_names, colnames(C))) stop("Latent matrix labels are inconsistent")
if (max(abs(C - t(C))) > 1e-10) stop("Latent matrix is not symmetric")
if (max(abs(diag(C) - 1)) > 1e-10) stop("Latent matrix does not have unit diagonal")

display <- setNames(labels$display_label, labels$metric)
if (any(is.na(display[metric_names]))) stop("Missing display labels")
rownames(C) <- display[metric_names]
colnames(C) <- display[metric_names]

# Required exact construction: Euclidean distances between rows of abs(C),
# followed by hclust's default complete linkage.
hc <- hclust(dist(abs(C)))

draw_tree <- function(path, device) {
  if (device == "png") {
    png(path, width = 4480, height = 2048, res = 320, bg = "white")
  } else {
    pdf(path, width = 14, height = 6.4, family = "Helvetica")
  }
  par(mar = c(4.0, 5, 4.5, 1.5), las = 2, cex = 1.04,
      cex.main = 1.45, cex.lab = 1.28, cex.axis = 1.08)
  plot(hc, main = paste0(args[[5]], ": latent-correlation profile clustering"),
       sub = "", xlab = "", ylab = "Height", hang = -1)
  mtext("Complete linkage of Euclidean distances between rows of |C|",
        side = 1, line = 2.0, cex = 1.05, las = 0)
  dev.off()
}
draw_tree(args[[3]], "png")
draw_tree(args[[4]], "pdf")

clusters <- vector("list", nrow(hc$merge))
for (i in seq_len(nrow(hc$merge))) {
  branch <- hc$merge[i, ]
  members <- unlist(lapply(branch, function(value) {
    if (value < 0) return(hc$labels[-value])
    strsplit(clusters[[value]]$members, " \\| ", fixed = FALSE)[[1]]
  }))
  clusters[[i]] <- list(
    members = paste(sort(members), collapse = " | "),
    size = length(members)
  )
}
merge_table <- data.frame(
  merge = seq_len(nrow(hc$merge)),
  height = hc$height,
  size = vapply(clusters, function(x) x$size, integer(1)),
  members = vapply(clusters, function(x) x$members, character(1)),
  stringsAsFactors = FALSE
)
write.csv(merge_table, args[[6]], row.names = FALSE)
