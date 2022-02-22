import sys
import os
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel

checkpoint_dir = sys.argv[1]
data_dir = sys.argv[2]
test_file = sys.argv[3]
out_file = sys.argv[4]

with open(test_file, 'r') as f:
    lines = f.read().splitlines()
    custom_lm = TransformerLanguageModel.from_pretrained(checkpoint_dir, data_name_or_path=data_dir, checkpoint_file='checkpoint_best.pt')
    lprobs = []
    count = 0
    perps = []
    for l in lines:
            if custom_lm.encode(l).size(0) > custom_lm.max_positions-2:
                l = ' '.join(l.split()[:custom_lm.max_positions-2]) 
            scores = custom_lm.score(l,shorten_method="truncate")['positional_scores']
            perps.append( scores.mean().neg().exp().item())
            lprobs.append(scores)
            
    print(checkpoint_dir, perps)        
    torch.save(lprobs, out_file)
