import sys

file_name = sys.argv[1]
out_name = sys.argv[2]
with  open(file_name, 'r') as f, open(out_name, 'w') as w:
    all_lines = f.read().splitlines()
    first = 0
    for l in all_lines:
        if 'INFO' not in l:
            first += 1
        else:
            break
    info_lines = [x for x in all_lines[first-1:] if 'INFO' not in x or 'Perplexity' in x]
    for i,j in zip(info_lines[::2], info_lines[1::2]):
        loss_ind = j.index("Loss (base 2):")+len("Loss (base 2):")
        perp_ind = j.index("Perplexity:")+len("Perplexity:")
        loss = j[loss_ind:loss_ind+j[loss_ind:].index(",")]
        perp = j[perp_ind:]
        w.write(','.join([i, loss, perp])+'\n')
