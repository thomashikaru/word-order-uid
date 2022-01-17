import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file, 'r') as f:
    lines = f.readlines()
    with open(out_file,'w') as out:
        for l in lines:
            words = l.rstrip().split()
            out.write(' '.join(words[::-1]) + '\n')

