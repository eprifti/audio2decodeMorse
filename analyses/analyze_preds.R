# Analyze combined manifest with predictions/loss and create plots.
# Usage (from repo root):
#   Rscript analyses/analyze_preds.R
#
# Adjust --input/--out-dir if your files live elsewhere:
#   Rscript analyses/analyze_preds.R --input analyses/combined_with_preds.csv --out-dir analyses/figures

# Ensure required packages are installed, then load them.
required_pkgs <- c("optparse", "readr", "dplyr", "ggplot2", "scales")
missing <- required_pkgs[!required_pkgs %in% rownames(installed.packages())]
if (length(missing) > 0) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
})

option_list <- list(
  make_option(c("-i", "--input"), default = "analyses/combined_with_preds.csv",
              help = "Path to combined_with_preds.csv"),
  make_option(c("-o", "--out-dir"), default = "analyses/figures",
              help = "Directory to write plots"),
  make_option(c("-c", "--char-errors"), default = "analyses/per_char_errors.csv",
              help = "Optional path to per_char_errors.csv (from per_char_errors.py) for alphabet-level plots"),
  make_option(c("-r", "--run-dir"), default = NULL,
              help = "Optional run directory (e.g., outputs/run1); overrides out-dir to <run-dir>/figures by default")
)
opts <- parse_args(OptionParser(option_list = option_list))

if (!is.null(opts$`run-dir`)) {
  opts$`out-dir` <- file.path(opts$`run-dir`, "figures")
  if (opts$`char-errors` == "analyses/per_char_errors.csv") {
    opts$`char-errors` <- file.path(opts$`run-dir`, "per_char_errors.csv")
  }
}

dir.create(opts$`out-dir`, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(opts$input, show_col_types = FALSE)

required_cols <- c("partition", "loss", "freq_hz", "amplitude", "wpm")
missing <- required_cols[!required_cols %in% names(df)]
if (length(missing) > 0) {
  stop(paste("Input is missing required columns:", paste(missing, collapse = ", ")))
}
has_text_len <- "text_len" %in% names(df)

# Precompute log-loss for models/plots (avoid log(0)).
df <- df %>% mutate(log_loss = log(loss + 1e-8))

# Basic summary statistics.
summary_stats <- df %>%
  group_by(partition) %>%
  summarise(
    n = n(),
    loss_mean = mean(loss, na.rm = TRUE),
    loss_median = median(loss, na.rm = TRUE),
    loss_sd = sd(loss, na.rm = TRUE),
    freq_mean = mean(freq_hz, na.rm = TRUE),
    amp_mean = mean(amplitude, na.rm = TRUE),
    wpm_mean = mean(wpm, na.rm = TRUE),
    text_len_mean = if (has_text_len) mean(text_len, na.rm = TRUE) else NA_real_,
    .groups = "drop"
  )

print("Summary by partition:")
print(summary_stats)

# Loss distribution by partition (log scale helps long tails).
p1 <- ggplot(df, aes(x = partition, y = loss, fill = partition)) +
  geom_boxplot(outlier.alpha = 0.25) +
  scale_y_log10(labels = label_number()) +
  labs(
    title = "CTC loss by partition",
    y = "Loss (log scale)", x = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(opts$`out-dir`, "loss_by_partition.png"), p1, width = 6, height = 4, dpi = 200)

# Loss vs frequency.
p2 <- ggplot(df, aes(x = freq_hz, y = loss, color = partition)) +
  geom_point(alpha = 0.2, size = 0.9) +
  geom_smooth(se = FALSE, method = "loess") +
  scale_y_log10(labels = label_number()) +
  labs(
    title = "Loss vs tone frequency",
    x = "Frequency (Hz)", y = "Loss (log scale)", color = "Partition"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(opts$`out-dir`, "loss_vs_frequency.png"), p2, width = 7, height = 4, dpi = 200)

# Loss vs amplitude (loudness proxy).
p3 <- ggplot(df, aes(x = amplitude, y = loss, color = partition)) +
  geom_point(alpha = 0.2, size = 0.9) +
  geom_smooth(se = FALSE, method = "loess") +
  scale_y_log10(labels = label_number()) +
  labs(
    title = "Loss vs amplitude",
    x = "Amplitude", y = "Loss (log scale)", color = "Partition"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(opts$`out-dir`, "loss_vs_amplitude.png"), p3, width = 7, height = 4, dpi = 200)

# Loss vs speed (WPM).
p4 <- ggplot(df, aes(x = wpm, y = loss, color = partition)) +
  geom_point(alpha = 0.2, size = 0.9) +
  geom_smooth(se = FALSE, method = "loess") +
  scale_y_log10(labels = label_number()) +
  labs(
    title = "Loss vs sending speed (WPM)",
    x = "Words per minute", y = "Loss (log scale)", color = "Partition"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(opts$`out-dir`, "loss_vs_wpm.png"), p4, width = 7, height = 4, dpi = 200)

# Simple linear model on log-loss.
lm_fit <- lm(log_loss ~ partition + freq_hz + amplitude + wpm, data = df)
lm_summary <- capture.output(summary(lm_fit))
writeLines(lm_summary, file.path(opts$`out-dir`, "linear_model_summary.txt"))
message("Linear model written to: ", file.path(opts$`out-dir`, "linear_model_summary.txt"))

# Loss density by partition.
p5 <- ggplot(df, aes(x = loss, color = partition, fill = partition)) +
  geom_density(alpha = 0.2) +
  scale_x_log10(labels = label_number()) +
  labs(
    title = "Loss density by partition",
    x = "Loss (log scale)", y = "Density", color = "Partition", fill = "Partition"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(opts$`out-dir`, "loss_density.png"), p5, width = 7, height = 4, dpi = 200)

# Frequency distribution by partition.
p6 <- ggplot(df, aes(x = freq_hz, fill = partition)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 40) +
  labs(
    title = "Tone frequency distribution",
    x = "Frequency (Hz)", y = "Count", fill = "Partition"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(opts$`out-dir`, "frequency_distribution.png"), p6, width = 7, height = 4, dpi = 200)

# Mean log-loss over freq/WPM grid (faceted by partition).
p7 <- ggplot(df, aes(x = freq_hz, y = wpm, z = log_loss)) +
  stat_summary_2d(fun = mean, bins = 35) +
  facet_wrap(~partition) +
  scale_fill_viridis_c(option = "C", name = "Mean log-loss") +
  labs(
    title = "Mean log-loss across frequency and speed",
    x = "Frequency (Hz)",
    y = "Words per minute"
  ) +
  theme_minimal(base_size = 12)
ggsave(file.path(opts$`out-dir`, "loss_freq_wpm_heatmap.png"), p7, width = 9, height = 5, dpi = 200)

# Correlation matrix for continuous vars.
cont_vars <- c("log_loss", "loss", "freq_hz", "amplitude", "wpm")
if (has_text_len) cont_vars <- c(cont_vars, "text_len")
cor_mat <- cor(df %>% select(any_of(cont_vars)), use = "complete.obs")
write.csv(cor_mat, file.path(opts$`out-dir`, "correlation_matrix.csv"))

# Loss vs text length (if available).
if (has_text_len) {
  p9 <- ggplot(df, aes(x = text_len, y = loss, color = partition)) +
    geom_point(alpha = 0.2, size = 0.9) +
    geom_smooth(se = FALSE, method = "loess") +
    scale_y_log10(labels = label_number()) +
    labs(
      title = "Loss vs text length",
      x = "Text length (characters)", y = "Loss (log scale)", color = "Partition"
    ) +
    theme_minimal(base_size = 12)
  ggsave(file.path(opts$`out-dir`, "loss_vs_text_length.png"), p9, width = 7, height = 4, dpi = 200)
} else {
  message("text_len column not found; skipping text length plots.")
}

# Alphabet-level error rates (if per_char_errors.csv exists).
char_err_path <- opts$`char-errors`
if (file.exists(char_err_path)) {
  char_df <- read_csv(char_err_path, show_col_types = FALSE)
  if (all(c("char", "error_rate", "count") %in% names(char_df))) {
    char_df <- char_df %>% arrange(desc(error_rate))
    p8 <- ggplot(char_df, aes(x = reorder(char, error_rate), y = error_rate, fill = count)) +
      geom_col() +
      coord_flip() +
      scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
      labs(
        title = "Per-character error rate",
        x = "Character",
        y = "Error rate",
        fill = "Count"
      ) +
      theme_minimal(base_size = 12)
    ggsave(file.path(opts$`out-dir`, "per_char_error_rates.png"), p8, width = 7, height = 6, dpi = 200)
  } else {
    warning("per_char_errors file missing required columns; skipping alphabet plot.")
  }
} else {
  message("per_char_errors file not found (", char_err_path, "); skipping alphabet-level plot.")
}

message("Wrote plots to: ", opts$`out-dir`)
