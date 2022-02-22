import sys
from mosestokenizer import MosesTokenizer
from string import punctuation

in_file = sys.argv[1]
out_file = sys.argv[2]
# list of languages can be found at http://www.loc.gov/standards/iso639-2/php/code_list.php 
language = sys.argv[3] if len(sys.argv) > 3 else 'en'


f = open(in_file, 'r')
with open(out_file,'w') as out,  MosesTokenizer(language) as tokenize:
    for line in f:
        if line.isspace():
            out.write('\n')
            continue
        sents = tokenize(line.rstrip())
        out.write(' '.join(sents) + '\n')
f.close()
