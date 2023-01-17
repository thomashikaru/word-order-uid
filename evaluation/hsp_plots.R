# generating plots for TACL submission

library(ggplot2)
library(dplyr)
library(tidyverse)
library(arrow)
library(ggpubr)

# order for showing variants and their corresponding colors
positions <- c("Real", "Reverse", "Approx", "Min-DL-Opt", "Random-1", "Random-2")
colors = c("#999999",
           "#D55E00",
           "#E69F00",
           "#66CC99",
           "#F0E442",
           "gold")

BASE_SIZE = 22
DATA_DIR = "/Users/thomasclark/mit/word-order-uid/evaluation/eval_results_cf_v6"

# make a point plot with facets for language, grammar variant on x-axis, and 
# a specified variable on the y-axis (e.g. mean surprisal)
make_point_plot <- function(csv_file, title, imgname, value, error, ncol) {
  data <- read.csv(paste(DATA_DIR, "plot_csv", csv_file, sep="/"))
  data[["min_ci"]] = data[[value]] - 1.96 * data[[error]]
  data[["max_ci"]] = data[[value]] + 1.96 * data[[error]]
  # data[["min_ci"]] = data[[value]] - data[[error]]
  # data[["max_ci"]] = data[[value]] + data[[error]]
  data_in <- data %>% 
    filter(variant %in% positions) %>% mutate(variant = factor(variant, levels=positions))
  print(nrow(data))
  print(nrow(data_in))
  
  plot <- ggplot(data=data_in, aes_string(x="variant", 
                              y=value, 
                              color="variant", 
                              ymin="min_ci", 
                              ymax="max_ci")) +
    geom_point(stat="identity", size=3) +
    geom_errorbar(stat="identity") + 
    facet_wrap(~language, scales="fixed", ncol=ncol) +
    ggtitle(title) +
    theme_light(base_size = BASE_SIZE) + 
    theme(legend.text=element_text(size=20), 
          strip.background = element_rect(fill="#f2f2f2"), 
          legend.position="right", 
          axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(), 
          strip.text = element_text(color="black", 
                                    margin = margin(1,0,1,0, "pt")),
          aspect.ratio = 0.7) +
    scale_x_discrete(limits = positions) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) +
    scale_color_manual(values = colors, name="Variant") +
    labs(x="", y="")
  ggsave(filename = paste(imgname, "_point.png", sep = ""), plot=plot,
         path = paste(DATA_DIR, "plots", sep="/"),
         width=12.75, height=7.5)
  ggsave(filename = paste(imgname, "_point.svg", sep = ""), plot=plot,
         path = paste(DATA_DIR, "plots", sep="/"),
         width=12.75, height=7.5)
  
  return(plot)
}

# FIGURES from paper

surprisal <- make_point_plot("surprisal_plot_vals.csv", "Mean Test \nSurprisal", 
                             "surprisal", "surprisalmean", "surprisalsem", 2)

variance <- make_point_plot("surprisal_variance_plot_vals.csv", 
                            "Mean Surprisal \nVariance", "surprisal_variance", 
                            "surprisalmean", "surprisalsem", 2)

mean_regress <- make_point_plot("surprisal_deviations_plot_vals.csv", 
                                "Surprisal Deviation \nfrom Dataset Mean", 
                                "surprisal_variance_dataset_mean", 
                                "surp_diff_squaredmean", "surp_diff_squaredsem", 2)

uid_power <- make_point_plot("infs_1.1_plot_vals.csv", "UID_power \n(k=1.1)", 
                             "uid_power_1.1", "surprisalmean", "surprisalsem", 2)

doc_initial <- make_point_plot("doc_initial_var.csv", 
                               "Mean Surprisal Variance \n(Doc Initial Sents)", 
                               "surprisal_variance_dataset_mean", 
                               "surprisalmean", "surprisalsem", 2)

uid_loc <- make_point_plot("delta_surps_plot_vals.csv", 
                           "Mean token-to-token \n\u0394surprisal", 
                           "delta_surp", "delta_surpmean", "delta_surpsem", 2)


# HSP Plot
ggarrange(surprisal, variance, mean_regress, uid_loc,
          labels = c("A", "B", "C", "D"),
          ncol = 4, common.legend = TRUE,
          legend="right", align="hv")

ggsave(filename = paste("hsp_fig2", ".png", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=20, height=8)


# SURPRISAL VARIANCE by language+variant (Doc Initial)
# first get surprisal variance for each sentence, then average across all sentences in dataset
make_point_plot("doc_initial_var.csv", "Surprisal Deviation from \nDataset Mean", "surprisal_variance_dataset_mean", "surprisalmean", "surprisalsem")
make_bar_plot("doc_initial_var.csv", "Surprisal Deviation from \nDataset Mean", "surprisal_variance_dataset_mean", "surprisalmean", "surprisalsem")

# SURPRISAL Deviation from dataset mean by language+variant
# mean deviation is computed for each sentence, then mean and std over all sentences in dataset
make_point_plot("surprisal_deviations_plot_vals.csv", "Mean Surprisal Variance \n(Doc Initial Sents)", "surprisal_variance_doc_initial", "surp_diff_squaredmean", "surp_diff_squaredsem")
make_bar_plot("surprisal_deviations_plot_vals.csv", "Mean Surprisal Variance \n(Doc Initial Sents)", "surprisal_variance_doc_initial", "surp_diff_squaredmean", "surp_diff_squaredsem")

# Average token-to-token delta surprisal
# mean delta_surprisal is computed for each sentence, then mean and std over all sentences in dataset
# note: the delta_surprisal is undefined for the first token in each document (since there is no previous token)
make_point_plot("delta_surps_plot_vals.csv", "Mean token-to-token \u0394surprisal", "delta_surp", "delta_surpmean", "delta_surpsem")
make_bar_plot("delta_surps_plot_vals.csv", "Mean token-to-token \u0394surprisal", "delta_surp", "delta_surpmean", "delta_surpsem")

# MAX surprisal
make_point_plot("max_surps_plot_vals.csv", "Maximum surprisal", "max_surp", "surprisalmean", "surprisalsem")
make_bar_plot("max_surps_plot_vals.csv", "Maximum surprisal", "max_surp", "surprisalmean", "surprisalsem")

# UID_power surprisal
make_point_plot("infs_plot_vals.csv", "UID_power (k=1.25)", "uid_power_1.25", "surprisalmean", "surprisalsem")
make_bar_plot("infs_plot_vals.csv", "UID_power (k=1.25)l", "uid_power_1.25", "surprisalmean", "surprisalsem")

make_point_plot("infs_1.1_plot_vals.csv", "UID_power (k=1.1)", "uid_power_1.1", "surprisalmean", "surprisalsem")
make_bar_plot("infs_1.1_plot_vals.csv", "UID_power (k=1.1)l", "uid_power_1.1", "surprisalmean", "surprisalsem")


# TOKEN-plot: Average surprisal
# the input dataframe has one row for each combination of (language, variant, sentence_len, sentence_pos)
# containing a mean and std over all surprisals fitting those criteria
# we could loop over different values of sentence_len
data <- read.csv(paste(DATA_DIR, "avg_surps_plot_vals.csv", sep="/"))
data %>% 
  filter(variant %in% positions_sub & sentence_len == 20) %>%
  mutate(variant = factor(variant, levels=positions_sub)) %>%
  ggplot(data=., aes(x=sentence_pos, y=surprisalmean, color=variant, 
                     ymin=surprisalmean - 1.96*surprisalsem, ymax=surprisalmean + 1.96*surprisalsem)) +
  geom_pointrange(alpha=0.3, size=0.5) + 
  geom_line(stat="smooth", alpha=0.5, size=1, span=0.5, se = FALSE) +
  facet_wrap(~language, scales="fixed", ncol=4) +
  ggtitle("Surprisal by token position") + 
  theme_light(base_size = BASE_SIZE) + 
  theme(aspect.ratio = 0.66) +
  scale_color_manual(values = colors_sub) +
  labs(x="token position in sentence", y="mean surprisal")
ggsave(filename = paste("surprisal_by_token_position.svg", sep = ""), 
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)
ggsave(filename = paste("surprisal_by_token_position.png", sep = ""), 
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)

# TOKEN-plot: Average change in surprisal
# the input dataframe has one row for each combination of (language, variant, sentence_len, sentence_pos)
# containing a mean and std over all delta_surprisals fitting those criteria
# note: the delta_surprisal is undefined for the first token in each document (since there is no previous token)
# we could loop over different values of sentence_len
data <- read.csv(paste(DATA_DIR, "delta_surps_by_tok.csv", sep="/"))
data %>% 
  filter(variant %in% positions_sub & sentence_len == 15) %>%
  mutate(variant = factor(variant, levels=positions_sub)) %>%
  ggplot(data=., aes(x=sentence_pos, y=delta_surpmean, color=variant, 
                     ymin=delta_surpmean - 1.96*delta_surpsem, ymax=delta_surpmean + 1.96*delta_surpsem)) +
  geom_pointrange(alpha=0.3, size=0.5) + 
  geom_line(stat="smooth", alpha=0.3, size=1, span=0.5, se = FALSE) +
  facet_wrap(~language, scales="fixed", ncol=4) +
  ggtitle("\u0394Surprisal by token position") + 
  theme_light(base_size = BASE_SIZE) + 
  scale_color_manual(values = colors_sub)
ggsave(filename = paste("delta_surprisal_by_token_position.svg", sep = ""), 
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=8.5)
ggsave(filename = paste("delta_surprisal_by_token_position.png", sep = ""), 
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=8.5)

