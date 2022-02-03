import sys
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from string import punctuation

in_file = sys.argv[1]
out_file = sys.argv[2]
# list of languages can be found at http://www.loc.gov/standards/iso639-2/php/code_list.php 
language = sys.argv[3] if len(sys.argv) > 3 else 'en'


f = open(in_file, 'r')
with open(out_file,'w') as out, MosesTokenizer(language) as tokenize, MosesSentenceSplitter(language) as splitsents:
    for line in f:        
        if line.isspace():
            out.write('\n')
            continue
        sents = splitsents([line])
        reversed_sents = []
        for sen in sents:
            words = tokenize(sen.rstrip())[::-1]
            punct_ind = None
            for i, w in enumerate(words):
                if w in ['.','?','!']:
                    punct_ind = i
                    break
            punct = words.pop(punct_ind) if punct_ind is not None else ''
            reversed_sents.append(' '.join(words + [punct]))
        out.write(' '.join(reversed_sents) + '\n')
f.close()
