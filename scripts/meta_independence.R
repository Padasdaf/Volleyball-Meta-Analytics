#!/usr/bin/env Rscript

abort <- function(...) {
  stop(sprintf(...), call. = FALSE)
}

read_integer_env <- function(name, default) {
  value <- Sys.getenv(name, unset = as.character(default))
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed) || parsed < 1L) {
    abort("%s must be a positive integer; got %s", name, value)
  }
  parsed
}

read_boolean_env <- function(name, default = TRUE) {
  value <- tolower(Sys.getenv(name, unset = if (default) "true" else "false"))
  if (value %in% c("true", "1", "yes")) return(TRUE)
  if (value %in% c("false", "0", "no")) return(FALSE)
  abort("%s must be true or false; got %s", name, value)
}

validate_probability <- function(value, label, tolerance = 1e-8) {
  if (length(value) != 1L || !is.finite(value)) {
    abort("%s is not a finite scalar", label)
  }
  if (value < -tolerance || value > 1 + tolerance) {
    abort("%s = %.16g lies materially outside [0, 1]", label, value)
  }
  # Only epsilon-level floating-point excursions are bounded.
  min(1, max(0, value))
}

independence_score <- function(C, metric, conditioning_set = character()) {
  metric_names <- colnames(C)
  if (is.null(metric_names) || !identical(rownames(C), metric_names)) {
    abort("C must have identical row and column metric names")
  }
  if (!(metric %in% metric_names)) {
    abort("Unknown target metric: %s", metric)
  }
  if (anyDuplicated(conditioning_set)) {
    abort("Conditioning set for %s contains duplicates", metric)
  }
  if (metric %in% conditioning_set) {
    abort("Target metric %s cannot condition on itself", metric)
  }
  unknown <- setdiff(conditioning_set, metric_names)
  if (length(unknown) > 0L) {
    abort("Unknown conditioning metrics for %s: %s", metric, paste(unknown, collapse = ", "))
  }
  if (length(conditioning_set) == 0L) {
    return(1)
  }

  conditioning_cov <- C[conditioning_set, conditioning_set, drop = FALSE]
  rhs <- C[conditioning_set, metric, drop = FALSE]
  solution <- tryCatch(
    solve(conditioning_cov, rhs),
    error = function(error) {
      abort(
        "Conditioning matrix for %s on {%s} is singular: %s",
        metric,
        paste(conditioning_set, collapse = ", "),
        conditionMessage(error)
      )
    }
  )
  score <- as.numeric(
    C[metric, metric] -
      C[metric, conditioning_set, drop = FALSE] %*% solution
  )
  validate_probability(score, sprintf("Independence(%s)", metric))
}

greedy_independence_curve <- function(C, target_metric, tie_tolerance = 1e-12) {
  current_set <- sort(setdiff(colnames(C), target_metric))
  rows <- list(data.frame(
    target_metric = target_metric,
    conditioning_size = length(current_set),
    independence = independence_score(C, target_metric, current_set),
    removed_metric = NA_character_,
    stringsAsFactors = FALSE
  ))

  while (length(current_set) > 0L) {
    candidates <- sort(current_set)
    candidate_scores <- vapply(
      candidates,
      function(candidate) {
        independence_score(C, target_metric, setdiff(current_set, candidate))
      },
      numeric(1)
    )
    best_score <- max(candidate_scores)
    tied <- candidates[abs(candidate_scores - best_score) <= tie_tolerance]
    removed <- sort(tied)[1]
    current_set <- setdiff(current_set, removed)
    rows[[length(rows) + 1L]] <- data.frame(
      target_metric = target_metric,
      conditioning_size = length(current_set),
      independence = independence_score(C, target_metric, current_set),
      removed_metric = removed,
      stringsAsFactors = FALSE
    )
  }

  curve <- do.call(rbind, rows)
  expected_sizes <- seq(ncol(C) - 1L, 0L, by = -1L)
  if (!identical(curve$conditioning_size, expected_sizes)) {
    abort("Greedy curve for %s has invalid conditioning sizes", target_metric)
  }
  if (abs(tail(curve$independence, 1L) - 1) > 1e-12) {
    abort("Greedy curve for %s does not end at independence 1", target_metric)
  }
  if (any(diff(curve$independence) < -1e-8)) {
    abort("Greedy curve for %s materially decreases after a removal", target_metric)
  }
  curve
}

run_mathematical_unit_tests <- function() {
  identity_C <- diag(4)
  dimnames(identity_C) <- list(LETTERS[1:4], LETTERS[1:4])
  for (metric in colnames(identity_C)) {
    score <- independence_score(identity_C, metric, setdiff(colnames(identity_C), metric))
    stopifnot(abs(score - 1) < 1e-12)
  }

  rho <- 0.6
  two_C <- matrix(c(1, rho, rho, 1), nrow = 2)
  dimnames(two_C) <- list(c("A", "B"), c("A", "B"))
  stopifnot(abs(independence_score(two_C, "A", "B") - (1 - rho^2)) < 1e-12)

  three_C <- matrix(
    c(1, 0.4, 0.2, 0.4, 1, 0.3, 0.2, 0.3, 1),
    nrow = 3,
    byrow = TRUE
  )
  dimnames(three_C) <- list(c("A", "B", "C"), c("A", "B", "C"))
  expected <- det(three_C) / det(three_C[c("B", "C"), c("B", "C")])
  actual <- independence_score(three_C, "A", c("B", "C"))
  stopifnot(abs(actual - expected) < 1e-12)

  curve <- greedy_independence_curve(three_C, "A")
  stopifnot(tail(curve$independence, 1L) == 1)
  stopifnot(all(diff(curve$independence) >= -1e-12))
  invisible(TRUE)
}

validate_correlation_matrix <- function(C, metric_names) {
  metric_count <- length(metric_names)
  if (!identical(dim(C), c(metric_count, metric_count))) {
    abort("C has dimensions %s; expected %d x %d", paste(dim(C), collapse = " x "), metric_count, metric_count)
  }
  if (!identical(rownames(C), metric_names) || !identical(colnames(C), metric_names)) {
    abort("C row and column names do not match input metric order")
  }
  if (any(!is.finite(C))) {
    abort("C contains non-finite entries")
  }

  symmetry_error <- max(abs(C - t(C)))
  diagonal_error <- max(abs(diag(C) - 1))
  if (symmetry_error > 1e-10) {
    abort("C is not symmetric: maximum error %.16g", symmetry_error)
  }
  if (diagonal_error > 1e-10) {
    abort("C diagonal differs from 1: maximum error %.16g", diagonal_error)
  }

  eigenvalues <- eigen(C, symmetric = TRUE, only.values = TRUE)$values
  eigen_tolerance <- 1e-10 * max(1, max(abs(eigenvalues)))
  if (min(eigenvalues) < -eigen_tolerance) {
    abort("C has a materially negative eigenvalue: %.16g", min(eigenvalues))
  }
  condition_number <- if (min(eigenvalues) > 0) {
    max(eigenvalues) / min(eigenvalues)
  } else {
    Inf
  }

  list(
    eigenvalues = eigenvalues,
    min_eigenvalue = min(eigenvalues),
    max_eigenvalue = max(eigenvalues),
    condition_number = condition_number,
    symmetry_error = symmetry_error,
    diagonal_error = diagonal_error
  )
}

positive_sequence_ess <- function(samples) {
  # sbgcop 0.975's summary.psgc sums a fixed block of noisy autocorrelations.
  # That legacy diagnostic can produce a negative denominator and therefore an
  # impossible negative ESS.  Retain it below for provenance, but report this
  # standard initial-positive-sequence diagnostic for the posterior C draws.
  sample_count <- length(samples)
  correlations <- as.numeric(acf(
    samples,
    lag.max = min(round(sample_count / 20), sample_count - 1L),
    plot = FALSE
  )$acf[-1])
  first_nonpositive <- which(correlations <= 0)[1]
  retained <- if (is.na(first_nonpositive)) {
    correlations
  } else if (first_nonpositive == 1L) {
    numeric()
  } else {
    correlations[seq_len(first_nonpositive - 1L)]
  }
  sample_count / (1 + 2 * sum(retained))
}

write_named_matrix <- function(matrix_value, first_column, path) {
  output <- data.frame(
    setNames(list(rownames(matrix_value)), first_column),
    as.data.frame(matrix_value, check.names = FALSE),
    check.names = FALSE
  )
  write.csv(output, path, row.names = FALSE, na = "NA")
}

plot_independence_curves <- function(curves, metric_names, output_path) {
  colors <- hcl.colors(length(metric_names), palette = "Dark 3")
  png(output_path, width = 3600, height = 2700, res = 300, type = "cairo")
  layout(matrix(c(1, 2), nrow = 1), widths = c(4.3, 1.2))
  par(mar = c(5.5, 5.5, 3.5, 1.5))
  plot(
    NA,
    xlim = c(0, length(metric_names) - 1L),
    ylim = c(0, 1),
    xaxs = "i",
    yaxs = "i",
    xaxt = "n",
    xlab = "Number of Metrics in Conditioning Set",
    ylab = "Independence",
    main = "Greedy Metric Independence Curves",
    cex.lab = 1.2,
    cex.main = 1.3
  )
  axis(1, at = 0:(length(metric_names) - 1L))
  grid(col = "gray85", lty = "dashed")
  for (index in seq_along(metric_names)) {
    metric_curve <- curves[curves$target_metric == metric_names[index], ]
    metric_curve <- metric_curve[order(metric_curve$conditioning_size), ]
    lines(
      metric_curve$conditioning_size,
      metric_curve$independence,
      col = colors[index],
      lwd = 2
    )
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend(
    "center",
    legend = metric_names,
    col = colors,
    lwd = 2,
    ncol = 1,
    bty = "n",
    cex = 0.88
  )
  dev.off()
}

plot_pca_variance <- function(pca_variance, output_path) {
  png(output_path, width = 2400, height = 1800, res = 300, type = "cairo")
  par(mar = c(5.5, 5.5, 3.5, 2))
  plot(
    pca_variance$component,
    pca_variance$cumulative_variance,
    type = "o",
    pch = 19,
    lwd = 2.5,
    xlim = c(1, nrow(pca_variance)),
    ylim = c(0, 1),
    xaxs = "i",
    yaxs = "i",
    xaxt = "n",
    xlab = "Number of Principal Components",
    ylab = "Cumulative Variance Explained",
    main = "Latent-Correlation PCA",
    cex.lab = 1.2,
    cex.main = 1.3
  )
  axis(1, at = pca_variance$component)
  grid(col = "gray85", lty = "dashed")
  dev.off()
}

plot_dependence_dendrogram <- function(C, output_path) {
  clustering <- hclust(dist(abs(C)))
  png(output_path, width = 3000, height = 2100, res = 300, type = "cairo")
  par(mar = c(9, 5, 4, 2))
  plot(
    clustering,
    main = "Latent Metric Correlation-Profile Clustering (Supplemental)",
    xlab = "Metric",
    sub = "Complete linkage; Euclidean distance between rows of |C|",
    ylab = "Euclidean Distance",
    hang = -1,
    cex = 0.9
  )
  dev.off()
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2L) {
    abort("Usage: Rscript meta_independence.R <input_csv> <output_dir>")
  }
  input_csv <- normalizePath(args[1], mustWork = TRUE)
  output_dir <- args[2]
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir, mustWork = TRUE)

  figure_dir <- Sys.getenv("INDEPENDENCE_FIGURE_DIR", unset = output_dir)
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  figure_dir <- normalizePath(figure_dir, mustWork = TRUE)
  make_figures <- read_boolean_env("INDEPENDENCE_MAKE_FIGURES", TRUE)
  seed <- read_integer_env("SBGCOP_SEED", 42L)
  nsamp <- read_integer_env("SBGCOP_NSAMP", 5000L)
  plugin_mode <- tolower(Sys.getenv("SBGCOP_PLUGIN_MODE", unset = "default"))
  if (!(plugin_mode %in% c("default", "full-rank"))) {
    abort("SBGCOP_PLUGIN_MODE must be default or full-rank; got %s", plugin_mode)
  }
  default_odens <- max(1L, round(nsamp / 1000))
  odens <- read_integer_env("SBGCOP_ODENS", default_odens)
  if (odens > nsamp) {
    abort("SBGCOP_ODENS cannot exceed SBGCOP_NSAMP")
  }
  if (floor(nsamp / odens) < 2L) {
    abort("MCMC settings must retain at least two posterior samples")
  }

  run_mathematical_unit_tests()
  cat("Mathematical independence tests passed.\n")

  expected_sbgcop_version <- Sys.getenv(
    "SBGCOP_EXPECTED_VERSION",
    unset = "0.975"
  )
  sbgcop_library <- Sys.getenv("SBGCOP_LIBRARY", unset = "")
  if (!nzchar(sbgcop_library)) {
    abort(
      paste0(
        "SBGCOP_LIBRARY is required. Use meta_independence.py so the exact ",
        "project-local sbgcop release is selected."
      )
    )
  }
  sbgcop_library <- normalizePath(sbgcop_library, mustWork = TRUE)
  .libPaths(c(sbgcop_library, .libPaths()))
  if (!requireNamespace("sbgcop", quietly = TRUE)) {
    abort("R package 'sbgcop' is not installed in %s", sbgcop_library)
  }
  actual_sbgcop_version <- as.character(utils::packageVersion("sbgcop"))
  actual_sbgcop_path <- normalizePath(find.package("sbgcop"), mustWork = TRUE)
  expected_sbgcop_path <- normalizePath(
    find.package("sbgcop", lib.loc = sbgcop_library),
    mustWork = TRUE
  )
  if (!identical(actual_sbgcop_version, expected_sbgcop_version)) {
    abort(
      "Wrong sbgcop version loaded: expected %s, got %s from %s",
      expected_sbgcop_version,
      actual_sbgcop_version,
      actual_sbgcop_path
    )
  }
  if (!identical(actual_sbgcop_path, expected_sbgcop_path)) {
    abort(
      "sbgcop was loaded outside the selected isolated library: %s",
      actual_sbgcop_path
    )
  }
  cat(sprintf("Verified sbgcop %s from %s.\n", actual_sbgcop_version, actual_sbgcop_path))

  input <- read.csv(
    input_csv,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN")
  )
  required_metadata <- c("season", "player_id")
  missing_metadata <- setdiff(required_metadata, names(input))
  if (length(missing_metadata) > 0L) {
    abort("Input is missing required columns: %s", paste(missing_metadata, collapse = ", "))
  }
  if (any(is.na(input[required_metadata]))) {
    abort("Input contains missing season or player_id values")
  }
  if (anyDuplicated(input[required_metadata])) {
    abort("Input contains duplicate (season, player_id) observations")
  }

  metric_names <- setdiff(names(input), required_metadata)
  if (length(metric_names) < 2L) {
    abort("Input must contain at least two metric columns")
  }
  for (metric in metric_names) {
    values <- input[[metric]]
    if (all(is.na(values))) {
      abort("Metric %s is entirely missing", metric)
    }
    if (!is.numeric(values)) {
      abort("Metric %s must be numeric", metric)
    }
    if (any(!is.finite(values) & !is.na(values))) {
      abort("Metric %s contains non-finite non-missing values", metric)
    }
    observed <- values[is.finite(values)]
    if (length(unique(observed)) < 2L) {
      abort("Metric %s is constant among observed values", metric)
    }
  }

  metric_matrix <- as.matrix(input[metric_names])
  storage.mode(metric_matrix) <- "double"
  rownames(metric_matrix) <- paste(input$season, input$player_id, sep = ":")

  input_population <- data.frame(
    season = sort(unique(input$season)),
    player_seasons = as.integer(table(input$season)[as.character(sort(unique(input$season)))]),
    stringsAsFactors = FALSE
  )
  write.csv(input_population, file.path(output_dir, "input_population.csv"), row.names = FALSE)

  input_missingness <- data.frame(
    metric = metric_names,
    observed = vapply(input[metric_names], function(values) sum(!is.na(values)), integer(1)),
    missing = vapply(input[metric_names], function(values) sum(is.na(values)), integer(1)),
    unique_observed = vapply(
      input[metric_names],
      function(values) length(unique(values[!is.na(values)])),
      integer(1)
    ),
    stringsAsFactors = FALSE
  )
  input_missingness$missing_fraction <- input_missingness$missing / nrow(input)
  write.csv(input_missingness, file.path(output_dir, "input_missingness.csv"), row.names = FALSE)

  plugin_threshold <- 100L
  package_default_plugin_marginal <- apply(metric_matrix, 2, function(values) {
    length(unique(values)) > plugin_threshold
  })
  # Retain the package default used by the Franks replication. With >100 unique
  # observed values, sbgcop treats a margin as continuous and plugs in its
  # empirical distribution while still sampling missing latent values and C.
  plugin_marginal <- if (plugin_mode == "default") {
    package_default_plugin_marginal
  } else {
    setNames(rep(FALSE, length(metric_names)), metric_names)
  }
  impute <- any(is.na(metric_matrix))

  # sbgcop.mcmc resets its own RNG from its seed argument, so both calls are
  # intentional. Other model controls retain the installed package defaults.
  set.seed(seed)
  fit_arguments <- list(
    Y = metric_matrix,
    nsamp = nsamp,
    odens = odens,
    seed = seed,
    verb = FALSE
  )
  if (plugin_mode == "full-rank") {
    # This is an explicit diagnostic mode. The primary Franks-style fit omits
    # plugin.marginal so that sbgcop applies its documented package default.
    fit_arguments$plugin.marginal <- plugin_marginal
  }
  elapsed <- system.time({
    fit <- do.call(sbgcop::sbgcop.mcmc, fit_arguments)
  })
  C <- apply(fit$C.psamp, c(1, 2), mean)
  dimnames(C) <- list(metric_names, metric_names)
  saveRDS(fit$C.psamp, file.path(output_dir, "C_posterior_samples.rds"))

  saved_samples <- dim(fit$C.psamp)[3]
  midpoint <- floor(saved_samples / 2)
  first_half_C <- apply(fit$C.psamp[, , seq_len(midpoint), drop = FALSE], c(1, 2), mean)
  second_half_C <- apply(
    fit$C.psamp[, , (midpoint + 1L):saved_samples, drop = FALSE],
    c(1, 2),
    mean
  )
  # sbgcop 0.975 exports summary.psgc but predates explicit S3 registration in
  # its NAMESPACE, so call the package implementation directly. Version 1.0
  # registers the same function for summary(fit).
  fit_summary <- getExportedValue("sbgcop", "summary.psgc")(fit)
  posterior_pair_ess <- vapply(
    seq_len(nrow(fit_summary$QC)),
    function(index) {
      pair <- strsplit(rownames(fit_summary$QC)[index], "*", fixed = TRUE)[[1]]
      positive_sequence_ess(fit$C.psamp[pair[1], pair[2], ])
    },
    numeric(1)
  )
  mcmc_diagnostics <- data.frame(
    statistic = c(
      "saved_samples",
      "min_effective_sample_size",
      "median_effective_sample_size",
      "max_effective_sample_size",
      "package_legacy_min_effective_sample_size",
      "package_legacy_median_effective_sample_size",
      "package_legacy_max_effective_sample_size",
      "max_first_vs_second_half_C_difference"
    ),
    value = c(
      saved_samples,
      min(posterior_pair_ess),
      median(posterior_pair_ess),
      max(posterior_pair_ess),
      min(fit_summary$ESS),
      median(fit_summary$ESS),
      max(fit_summary$ESS),
      max(abs(first_half_C - second_half_C))
    )
  )
  write.csv(
    mcmc_diagnostics,
    file.path(output_dir, "mcmc_diagnostics.csv"),
    row.names = FALSE
  )

  validation <- validate_correlation_matrix(C, metric_names)
  write_named_matrix(C, "metric", file.path(output_dir, "latent_correlation.csv"))
  validation_table <- data.frame(
    statistic = c(
      "min_eigenvalue",
      "max_eigenvalue",
      "condition_number",
      "max_symmetry_error",
      "max_diagonal_error"
    ),
    value = c(
      validation$min_eigenvalue,
      validation$max_eigenvalue,
      validation$condition_number,
      validation$symmetry_error,
      validation$diagonal_error
    )
  )
  write.csv(
    validation_table,
    file.path(output_dir, "latent_correlation_validation.csv"),
    row.names = FALSE
  )

  full_scores <- vapply(
    metric_names,
    function(metric) independence_score(C, metric, setdiff(metric_names, metric)),
    numeric(1)
  )
  independence_scores <- data.frame(
    metric = metric_names,
    independence = as.numeric(full_scores),
    explained_fraction = 1 - as.numeric(full_scores),
    stringsAsFactors = FALSE
  )
  independence_scores <- independence_scores[
    order(-independence_scores$independence, independence_scores$metric),
  ]
  rownames(independence_scores) <- NULL
  write.csv(
    independence_scores,
    file.path(output_dir, "independence_scores.csv"),
    row.names = FALSE
  )

  independence_curves <- do.call(
    rbind,
    lapply(metric_names, function(metric) greedy_independence_curve(C, metric))
  )
  rownames(independence_curves) <- NULL
  write.csv(
    independence_curves,
    file.path(output_dir, "independence_curves.csv"),
    row.names = FALSE,
    na = "NA"
  )

  pca <- eigen(C, symmetric = TRUE)
  eigenvalue_sum <- sum(pca$values)
  if (abs(eigenvalue_sum - length(metric_names)) > 1e-8) {
    abort(
      "PCA eigenvalues sum to %.16g, expected %d",
      eigenvalue_sum,
      length(metric_names)
    )
  }
  if (min(pca$values) < -1e-10) {
    abort("PCA contains a materially negative eigenvalue: %.16g", min(pca$values))
  }
  pca_variance <- data.frame(
    component = seq_along(pca$values),
    eigenvalue = pca$values,
    variance_fraction = pca$values / eigenvalue_sum,
    cumulative_variance = cumsum(pca$values) / eigenvalue_sum
  )
  if (abs(tail(pca_variance$cumulative_variance, 1L) - 1) > 1e-10) {
    abort("PCA cumulative variance does not end at 1")
  }
  write.csv(pca_variance, file.path(output_dir, "pca_variance.csv"), row.names = FALSE)

  startup_sensitivity <- do.call(
    rbind,
    lapply(c(0, 0.1, 0.2), function(discard_fraction) {
      first_sample <- floor(saved_samples * discard_fraction) + 1L
      retained <- first_sample:saved_samples
      sensitivity_C <- apply(
        fit$C.psamp[, , retained, drop = FALSE],
        c(1, 2),
        mean
      )
      dimnames(sensitivity_C) <- list(metric_names, metric_names)
      sensitivity_scores <- vapply(
        metric_names,
        function(metric) {
          independence_score(
            sensitivity_C,
            metric,
            setdiff(metric_names, metric)
          )
        },
        numeric(1)
      )
      sensitivity_eigenvalues <- eigen(
        sensitivity_C,
        symmetric = TRUE,
        only.values = TRUE
      )$values
      sensitivity_cumulative <- cumsum(sensitivity_eigenvalues) /
        sum(sensitivity_eigenvalues)
      data.frame(
        discarded_fraction = discard_fraction,
        retained_samples = length(retained),
        max_C_difference_from_all = max(abs(sensitivity_C - C)),
        max_score_difference_from_all = max(abs(sensitivity_scores - full_scores)),
        max_pca_cumulative_difference_from_all = max(
          abs(sensitivity_cumulative - pca_variance$cumulative_variance)
        )
      )
    })
  )
  write.csv(
    startup_sensitivity,
    file.path(output_dir, "mcmc_startup_sensitivity.csv"),
    row.names = FALSE
  )

  pca_loadings <- pca$vectors
  rownames(pca_loadings) <- metric_names
  colnames(pca_loadings) <- paste0("PC", seq_len(ncol(pca_loadings)))
  write_named_matrix(
    pca_loadings,
    "metric",
    file.path(output_dir, "pca_loadings.csv")
  )

  metadata <- data.frame(
    setting = c(
      "R_version",
      "sbgcop_version",
      "seed",
      "nsamp",
      "odens",
      "saved_samples",
      "S0",
      "n0",
      "impute",
      "plugin_threshold",
      "plugin_mode",
      "plugin_marginal_used",
      "package_default_plugin_marginal_metrics",
      "plugin_marginal_metrics",
      "player_seasons",
      "metrics",
      "elapsed_seconds"
    ),
    value = c(
      R.version.string,
      actual_sbgcop_version,
      seed,
      nsamp,
      odens,
      floor(nsamp / odens),
      "identity",
      ncol(metric_matrix) + 2L,
      impute,
      plugin_threshold,
      plugin_mode,
      if (plugin_mode == "default") "package default" else "explicit all FALSE",
      paste(
        names(package_default_plugin_marginal)[package_default_plugin_marginal],
        collapse = ";"
      ),
      paste(names(plugin_marginal)[plugin_marginal], collapse = ";"),
      nrow(metric_matrix),
      ncol(metric_matrix),
      unname(elapsed["elapsed"])
    ),
    stringsAsFactors = FALSE
  )
  write.csv(metadata, file.path(output_dir, "analysis_metadata.csv"), row.names = FALSE)

  if (make_figures) {
    plot_independence_curves(
      independence_curves,
      metric_names,
      file.path(figure_dir, "Figure_2_Independence.png")
    )
    plot_pca_variance(
      pca_variance,
      file.path(figure_dir, "Figure_9_PCA.png")
    )
    plot_dependence_dendrogram(
      C,
      file.path(figure_dir, "Figure_4_Dendrogram.png")
    )
  }

  cat("\nLatent correlation validation:\n")
  print(validation_table, row.names = FALSE)
  cat("\nFull-set independence (most to least independent):\n")
  print(independence_scores, row.names = FALSE, digits = 6)
  cat(sprintf("\nCompleted sbgcop analysis in %.3f seconds.\n", unname(elapsed["elapsed"])))
}

main()
