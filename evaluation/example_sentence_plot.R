# Make example sentence plot - 
# show same sentence in different variants with per-token surprisals

library(ggplot2)
library(dplyr)

# subset of variants to plot
variants = c("Real", "Reverse", "Random-1", "Efficient-OV")

# appearance
BASE_SIZE = 22

# input data
DATA_DIR = "/Users/thomasclark/mit/word-order-uid/evaluation/eval_results_cf_v6"

# helper function (a somewhat hacky way to get axis to present in right order)
f <- function(x) sub("[^_]*_[^_]*_","",x) 

data <- read.csv(paste(DATA_DIR, "english_100_docs.csv", sep="/"))

df <- data %>%
  filter(document_id == 3, sentence_id == 7, variant %in% variants) %>%
  select(document_id, sentence_id, variant, token, surprisal, sentence_pos) %>%
  unique() %>%
  mutate(variant = factor(variant, levels=variants))

                     
cbind(df, id=paste(df$variant, df$sentence_pos, df$token,sep="_")) %>%
ggplot(., aes(y = surprisal, x=reorder(id, sentence_pos), group=1)) +
  geom_point() +
  facet_wrap(~variant, scales="free_x") +
  scale_x_discrete(labels=f) +
  geom_line(color="blue") +
  labs(x="token", y="surprisal") +
  theme_light(base_size = BASE_SIZE) +
  theme(axis.text.x = element_text(angle=45, size=15, hjust=1, vjust=1),
        strip.text = element_text(color="black", margin = margin(1,0,1,0, "pt")),
        strip.background = element_rect(fill="#e9f7f6"))
ggsave(filename = paste("example_sentence.svg", sep = ""), 
       path = DATA_DIR,
       width=12.75, height=8.50)
ggsave(filename = paste("example_sentence.png", sep = ""), 
       path = DATA_DIR,
       width=12.75, height=8.50)
