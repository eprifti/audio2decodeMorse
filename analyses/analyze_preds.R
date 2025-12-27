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
              help = "Directory to write plots")
)
opts <- parse_args(OptionParser(option_list = option_list))

dir.create(opts$`out-dir`, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(opts$input, show_col_types = FALSE)

if (!all(c("partition", "loss", "freq_hz", "amplitude", "wpm") %in% names(df))) {
  stop("Input is missing required columns: partition, loss, freq_hz, amplitude, wpm")
}

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
df <- df %>% mutate(log_loss = log(loss + 1e-8))
lm_fit <- lm(log_loss ~ partition + freq_hz + amplitude + wpm, data = df)
lm_summary <- capture.output(summary(lm_fit))
writeLines(lm_summary, file.path(opts$`out-dir`, "linear_model_summary.txt"))
message("Linear model written to: ", file.path(opts$`out-dir`, "linear_model_summary.txt"))

message("Wrote plots to: ", opts$`out-dir`)
