#!/usr/bin/env Rscript

## Hass & Craig (2018) coefficient estimator.  The paper identifies GLMNET
## Ridge as the preferred coefficient-only implementation but does not publish
## its call, lambda choice, folds, or package version.  We use the
## contemporaneous CRAN release.  The paper does not publish the CV folds,
## seed, lambda rule, standardization switch, or treatment of the home-context
## coefficient.  The explicit conventional choices below are therefore source-
## specification assumptions, not claims about unpublished author code.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: advanced_apm.R INPUT.csv OUTPUT.csv")

extra_lib <- Sys.getenv("ADVANCED_R_LIB", unset = "")
if (nzchar(extra_lib)) .libPaths(c(extra_lib, .libPaths()))
if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("pinned glmnet 2.0-16 is required; run setup_advanced_r_environment.py")
}
if (utils::packageDescription("glmnet")$Version != "2.0-16") {
    stop("APM requires pinned glmnet 2.0-16")
}

input <- utils::read.csv(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
if (!all(c("outcome", "sample_weight", "cv_fold_id") %in% names(input))) {
    stop("APM input lacks outcome/sample_weight/cv_fold_id")
}
if (!"home_context" %in% names(input)) stop("APM input lacks home_context")
player_ids <- setdiff(
    names(input), c("outcome", "sample_weight", "home_context", "cv_fold_id")
)
if (!length(player_ids)) stop("APM input has no player columns")
x <- as.matrix(input[, c("home_context", player_ids), drop = FALSE])
y <- input$outcome
w <- input$sample_weight
if (length(unique(y)) != 2L) stop("APM response must contain both outcomes")

set.seed(42)
if (nrow(x) %% 2L != 0L) stop("APM complementary focal-team rows are unpaired")
fold_id <- as.integer(input$cv_fold_id)
if (any(!is.finite(fold_id)) || any(fold_id < 1L) || length(unique(fold_id)) != 10L) {
    stop("APM CV fold IDs must contain ten positive integer folds")
}
if (any(fold_id[seq(1L, length(fold_id), by = 2L)] !=
        fold_id[seq(2L, length(fold_id), by = 2L)])) {
    stop("Complementary views of one physical point must share a CV fold")
}
fitted <- glmnet::cv.glmnet(
    x = x,
    y = y,
    weights = w,
    family = "binomial",
    alpha = 0,
    nfolds = 10,
    foldid = fold_id,
    type.measure = "deviance",
    standardize = TRUE,
    intercept = TRUE,
    penalty.factor = c(0, rep(1, length(player_ids)))
)
coefs <- as.matrix(stats::coef(fitted, s = "lambda.min"))[, 1]
effects <- unname(coefs[player_ids])
output <- data.frame(
    player_id = player_ids,
    log_odds_effect = effects,
    APM_per_50 = 50 * stats::plogis(effects) - 25,
    model_intercept = unname(coefs[["(Intercept)"]]),
    home_context_effect = unname(coefs[["home_context"]]),
    lambda_min = fitted$lambda.min,
    lambda_1se = fitted$lambda.1se,
    stringsAsFactors = FALSE
)
utils::write.csv(output, args[[2]], row.names = FALSE, na = "")
