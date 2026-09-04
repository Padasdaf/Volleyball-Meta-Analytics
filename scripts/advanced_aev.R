#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: advanced_aev.R INPUT.csv OUTPUT.csv")
extra_lib <- Sys.getenv("ADVANCED_R_LIB", unset = "")
if (nzchar(extra_lib)) .libPaths(c(extra_lib, .libPaths()))
if (!requireNamespace("ovlytics", quietly = TRUE)) {
    stop("pinned ovlytics is required; run setup_advanced_r_environment.py")
}
description <- utils::packageDescription("ovlytics")
expected_sha <- "96d0670d0f9fcc856bffb4b6182314279f4c4a6b"
if (description$Version != "0.3.3" || description$RemoteSha != expected_sha) {
    stop("AEV requires openvolley/ovlytics 0.3.3 at the pinned source commit")
}

x <- utils::read.csv(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
id_columns <- grep("(_id$|_player_id[1-6]$)", names(x), value = TRUE)
for (column in id_columns) x[[column]] <- as.character(x[[column]])
x$player_name <- as.character(x$player_name)
x$home_setter_position <- as.integer(x$home_setter_position)
x$visiting_setter_position <- as.integer(x$visiting_setter_position)

result <- ovlytics::ov_aev(
    x,
    rotation = "SHM",
    calculate_by = "match_id",
    report_by = "setter_id",
    min_N_attacks = 10,
    detail = FALSE
)
utils::write.csv(result, args[[2]], row.names = FALSE, na = "")
