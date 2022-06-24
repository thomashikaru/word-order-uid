# Word Order UID Project

This repo is forked from fairseq. This README contains information specific to our fork, which involves experiments probing the relationship between word order and Uniform Information Density. 

## File Overview

- `preprocess*`: These scripts take datsets and put them in the correct format for training models using the fairseq CLI. 
- `eval*`: These scripts take trained models and evaluate them on their test sets, generally outputting per-word surprisals/logprobs for sentences/documents in the test set. 
- `apply_bpe*` and `apply_sentencepiece*`: These scripts take a plain text dataset and apply byte pair encoding or sentencepiece encoding. These are tokenization strategies that handle out-of-vocabulary tokens by breaking them into known word pieces or sentence pieces. 
- `train*`: These scripts train language models. There are a variety of architectures, such as LSTM, vanilla transformer, and seq2seq (or encoder-decoder) models. 

Additionally, there are scripts for downloading the `wiki40b` dataset and for downsampling the dataset.


## Steps for reproducing counterfactual LM comparison

1. Generate counterfactual data
1. Train BPE on all data
1. Apply BPE to counterfactual data
1. Preprocess data (fairseq-preprocess)
1. Train LMs (fairseq-train)
1. Evaluate LMs (custom script)
1. Perform Analysis