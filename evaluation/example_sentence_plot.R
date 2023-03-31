# Make example sentence plot - 
# show same sentence in different variants with per-token surprisals

library(ggplot2)
library(dplyr)

# subset of variants to plot
variants = c("Real", "Reverse", "Random-1", "Efficient-OV", "Sort-Freq")

# appearance
BASE_SIZE = 22

# input data
BASE_DIR = "/Users/thomasclark/mit/word-order-uid"
DATA_DIR = paste(BASE_DIR, "evaluation/perps-cf-diff-sizes/20000000/1", sep="/")

# helper function (a somewhat hacky way to get axis to present in right order)
f <- function(x) {
  sub("[^_]*_[^_]*_","",x) 
}

data.1 <- read.csv(paste(DATA_DIR, "en-REAL_REAL_example_sentences.csv", sep="/"))
data.2 <- read.csv(paste(DATA_DIR, "en-REVERSE_example_sentences.csv", sep="/"))
data.3 <- read.csv(paste(DATA_DIR, "en-RANDOM_1_example_sentences.csv", sep="/"))
data.4 <- read.csv(paste(DATA_DIR, "en-SORT_FREQ_example_sentences.csv", sep="/"))
data <- rbind(data.1, data.2, data.3, data.4)


df <- data %>%
  filter(document_id == 3, sentence_id == 7, variant %in% variants) %>%
  select(document_id, sentence_id, variant, token, surprisal, sentence_pos) %>%
  unique() %>%
  mutate(variant = factor(variant, levels=variants))

                     
cbind(df, id=paste(df$variant, df$sentence_pos, df$token,sep="_")) %>%
ggplot(., aes(y = surprisal, x=reorder(id, sentence_pos), group=1)) +
  facet_wrap(~variant, scales="free_x") +
  scale_x_discrete(labels=f) +
  geom_line(color="blue", size=2) +
  geom_point(size=3, color="red") +
  labs(x="token", y="surprisal") +
  theme_light(base_size = BASE_SIZE) +
  theme(axis.text.x = element_text(angle=45, size=15, hjust=1, vjust=1),
        strip.text = element_text(color="black", margin = margin(1,0,1,0, "pt"), size=20),
        strip.background = element_rect(fill="#e9f7f6"),
        aspect.ratio=0.2, text=element_text(family="serif"))

# save to files
ggsave(filename = paste("example_sentence.svg", sep = ""), 
       path = DATA_DIR,
       width=12, height=5)
ggsave(filename = paste("example_sentence.png", sep = ""), 
       path = DATA_DIR,
       width=12, height=5)
