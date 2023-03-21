# generating plots for TACL submission
library(ggplot2)
library(dplyr)
library(arrow)
library(ggpubr)

# list of pseudo-grammars
pseudo_grams = c("Real", "Reverse", "Sort-Freq", "Sort-Freq-Rev", "Min-DL-Loc")

# order for showing variants and their corresponding colors
positions <- c("Real", "Reverse", "Sort-Freq", "Sort-Freq-Rev", "Approx", "Min-DL-Loc", 
               "Min-DL-Opt", "Efficient-OV", "Efficient-VO", "Random-1", "Random-2", "Random-3", "Random-4", "Random-5")
positions_new <- c("Real", "Reverse", "Sort_Freq", "Sort_Freq_Rev", "Approx", "Min_DL_Loc", 
                   "Min_DL_Opt", "Efficient_OV", "Efficient_VO", "Random_1", "Random_2", "Random_3", "Random_4", "Random_5")

colors = c("#999999",
           "#D55E00",
           "#56B4E9", 
           "#0072B2", 
           "#E69F00",
           "#009E73",
           "#66CC99",
           "#CC79A7", 
           "#9999CC",
           "gold",
           "gold",
           "gold", 
           "gold", 
           "gold")

BASE_SIZE = 22

# make a point plot with facets for language, grammar variant on x-axis, and 
# a specified variable on the y-axis (e.g. mean surprisal)
make_point_plot <- function(data, title, imgname, value, error, ncol) {
  data[["min_ci"]] = data[[value]] - 1.96 * data[[error]]
  data[["max_ci"]] = data[[value]] + 1.96 * data[[error]]
  plot <- data %>% 
    filter(variant %in% positions_new) %>%
    mutate(variant = factor(variant, levels=positions_new),
           class = variant %in% pseudo_grams) %>%
    ggplot(data=., aes_string(x="variant", 
                            y=value, 
                            color="variant", 
                            ymin="min_ci", 
                            ymax="max_ci")) +
    geom_point(stat="identity", size=2) +
    geom_errorbar(stat="identity") + 
    facet_wrap(~language, scales="fixed", ncol=ncol) +
    ggtitle(title) +
    theme_light(base_size = BASE_SIZE) + 
    theme(legend.text=element_text(size=16), 
          strip.background = element_rect(fill="#f2f2f2"), 
          legend.position="right", 
          axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(), 
          strip.text = element_text(color="black", 
                                   margin = margin(1,0,1,0, "pt")),
          aspect.ratio = 0.4) +
    scale_x_discrete(limits = positions_new) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 3)) +
    scale_color_manual(values = colors, name="Variant") +
    labs(x="", y="")
  # ggsave(filename = paste(imgname, "_point.png", sep = ""), 
  #        path = paste(DATA_DIR, "plots", sep="/"),
  #        width=12.75, height=7.5)
  # ggsave(filename = paste(imgname, "_point.svg", sep = ""), 
  #        path = paste(DATA_DIR, "plots", sep="/"),
  #        width=12.75, height=7.5)
  
  return(plot)
}

# FIGURES from paper

# joint plot for surprisal and surprisal variance
BASE_DIR = "/Users/thomasclark/mit/word-order-uid"
DATA_DIR = paste(BASE_DIR, "evaluation/perps-cf-diff-sizes", sep="/")
data <- read.csv(paste(DATA_DIR, "results_summary.csv", sep="/"))
data <- data %>% filter(num_toks == 20000000)

DATA_DIR_CC = paste(BASE_DIR, "evaluation/cc100/perps-cf")
data.cc <- read.csv(paste(DATA_DIR_CC, "results_summary.csv", sep="/"))

data$surp_var_div_mean = data$surprisal_var_mean / data$surprisal_mean

p <- make_point_plot(data, "Suprisal Variance (div by Mean Surp)", "surp_var_div_mean", "surp_var_div_mean", "surprisal_var_sem", 2)
p

surprisal <- make_point_plot(data, "Mean Test Surprisal", "surprisal", "surprisal_mean", "surprisal_sem", 2)
variance <- make_point_plot(data, "Mean Surprisal Variance", "surprisal_variance", "surprisal_var_mean", "surprisal_var_sem", 2)

ggarrange(surprisal, variance,
          labels = c("A", "B"),
          ncol = 2, common.legend = TRUE,
          legend="right", align="hv")

ggsave(filename = paste("joint_surprisal_and_variance", ".png", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)
ggsave(filename = paste("joint_surprisal_and_variance", ".svg", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)

variance.cc <- make_point_plot(data.cc, "Mean Surprisal Variance\n(CC100 Dataset)", "surprisal_variance_cc100", "surprisal_var_mean", "surprisal_var_sem", 1)
ggsave(filename = paste("suprisal_variance_cc100", ".png", sep = ""),
       path = paste(DATA_DIR_CC, "plots", sep="/"),
       width=8, height=10)

surprisal_var_norm <- make_point_plot(data, "Surprisal Variance (Normalized)", "surprisal_var_norm", "surprisal_var_norm_mean", "surprisal_var_norm_sem", 2)

# joint plot for surprisal variance (dataset mean surprisal) and UID_power metric
mean_regress <- make_point_plot(data, "Surprisal Deviation \nfrom Dataset Mean", "surprisal_variance_dataset_mean", "surp_diff_squared_mean", "surp_diff_squared_sem", 2)
uid_power <- make_point_plot(data, "UID_power \n(k=1.25)", "uid_power_1.25", "uidp_1_25_mean", "uidp_1_25_sem", 2)

ggarrange(mean_regress, uid_power,
          labels = c("A", "B"),
          ncol = 2, common.legend = TRUE,
          legend="right", align="hv")

ggsave(filename = paste("joint_mean_regress_and_uid_power", ".png", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)
ggsave(filename = paste("joint_mean_regress_and_uid_power", ".svg", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)


# joint plot for doc-initial surprisal variance and UID_loc metric
doc_initial <- make_point_plot(data, "Mean Surprisal Variance \n(Doc Initial Sents)", "surprisal_variance_doc_initial", "surprisal_var_doc_initial_mean", "surprisal_var_doc_initial_sem", 2)
uid_loc <- make_point_plot(data, "Mean token-to-token \n\u0394surprisal", "delta_surp", "delta_surp_mean", "delta_surp_sem", 2)

ggarrange(doc_initial, uid_loc,
          labels = c("A", "B"),
          ncol = 2, common.legend = TRUE,
          legend="right", align="hv")

ggsave(filename = paste("joint_doc_initial_and_uid_loc", ".png", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)
ggsave(filename = paste("joint_doc_initial_and_uid_loc", ".svg", sep = ""),
       path = paste(DATA_DIR, "plots", sep="/"),
       width=12.75, height=7)


# TOKEN-plot: Average surprisal
# the input dataframe has one row for each combination of (language, variant, sentence_len, sentence_pos)
# containing a mean and std over all surprisals fitting those criteria
# we could loop over different values of sentence_len
# data <- read.csv(paste(DATA_DIR, "plot_csv", "avg_surps.csv", sep="/"))
# data %>% 
#   filter(variant %in% positions_sub & sentence_len == 20) %>%
#   mutate(variant = factor(variant, levels=positions_sub)) %>%
#   ggplot(data=., aes(x=sentence_pos, y=surprisalmean, color=variant, 
#                      ymin=surprisalmean - 1.96*surprisalsem, ymax=surprisalmean + 1.96*surprisalsem)) +
#   geom_pointrange(alpha=0.3, size=0.5) + 
#   geom_line(stat="smooth", alpha=0.5, size=1, span=0.5, se = FALSE) +
#   facet_wrap(~language, scales="fixed", ncol=4) +
#   ggtitle("Surprisal by token position") + 
#   theme_light(base_size = BASE_SIZE) + 
#   theme(aspect.ratio = 0.66) +
#   scale_color_manual(values = colors_sub) +
#   labs(x="token position in sentence", y="mean surprisal")
# ggsave(filename = paste("surprisal_by_token_position.svg", sep = ""), 
#        path = paste(DATA_DIR, "plots", sep="/"),
#        width=12.75, height=7)
# ggsave(filename = paste("surprisal_by_token_position.png", sep = ""), 
#        path = paste(DATA_DIR, "plots", sep="/"),
#        width=12.75, height=7)
# 
# # TOKEN-plot: Average change in surprisal
# # the input dataframe has one row for each combination of (language, variant, sentence_len, sentence_pos)
# # containing a mean and std over all delta_surprisals fitting those criteria
# # note: the delta_surprisal is undefined for the first token in each document (since there is no previous token)
# # we could loop over different values of sentence_len
# data <- read.csv(paste(DATA_DIR, "plot_csv", "delta_surps_by_tok.csv", sep="/"))
# data %>% 
#   filter(variant %in% positions_sub & sentence_len == 15) %>%
#   mutate(variant = factor(variant, levels=positions_sub)) %>%
#   ggplot(data=., aes(x=sentence_pos, y=delta_surpmean, color=variant, 
#                      ymin=delta_surpmean - 1.96*delta_surpsem, ymax=delta_surpmean + 1.96*delta_surpsem)) +
#   geom_pointrange(alpha=0.3, size=0.5) + 
#   geom_line(stat="smooth", alpha=0.3, size=1, span=0.5, se = FALSE) +
#   facet_wrap(~language, scales="fixed", ncol=4) +
#   ggtitle("\u0394Surprisal by token position") + 
#   theme_light(base_size = BASE_SIZE) + 
#   scale_color_manual(values = colors_sub)
# ggsave(filename = paste("delta_surprisal_by_token_position.svg", sep = ""), 
#        path = paste(DATA_DIR, "plots", sep="/"),
#        width=12.75, height=8.5)
# ggsave(filename = paste("delta_surprisal_by_token_position.png", sep = ""), 
#        path = paste(DATA_DIR, "plots", sep="/"),
#        width=12.75, height=8.5)

