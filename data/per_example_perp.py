# This script loads a trained fairseq language model and evaluates it on a test set
# Command-line arguments:
# 1. directory with checkpoint. assumes the dir contains a checkpoint_best.pt
# 2. directory with test data. expects the result of fairseq-preprocess (bin data)
# 3. file with test data. expects plain text with bpe
# 4. output file where logprobs and tokens will be saved


import sys
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel

checkpoint_dir = sys.argv[1]
data_dir = sys.argv[2]
test_file = sys.argv[3]
out_file = sys.argv[4]

with open(test_file, "r") as f:
    lines = f.read().splitlines()
    custom_lm = TransformerLanguageModel.from_pretrained(
        checkpoint_dir, data_name_or_path=data_dir, checkpoint_file="checkpoint_best.pt"
    )
    lprobs = []
    count = 0
    perps = []
    tokens = []
    for l in lines:
        if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
            l = " ".join(l.split()[: custom_lm.max_positions - 2])
        out = custom_lm.score(l, shorten_method="truncate")
        perps.append(out["positional_scores"].mean().neg().exp().item())
        lprobs.append(out["positional_scores"])
        tokens.append([custom_lm.tgt_dict[i] for i in out["tokens"]])
    print(checkpoint_dir, perps)
    torch.save([lprobs, tokens], out_file)

