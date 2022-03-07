import sys
import argparse

from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from string import punctuation

def tokenize_strings(f, out, lang, split_sentences=False, reverse=False, document_level_processing=False):
    with MosesTokenizer(lang) as tokenize, MosesSentenceSplitter(lang) as splitsents:
        for line in f:        
            if line.isspace():
                out.write('\n')
                continue
            if document_level_processing:
                words = tokenize(line.rstrip())
                words = words[::-1] if reverse else words
                out.write(' '.join(words) + '\n')
            else:
                sents = splitsents([line])
                new_sents = []
                for sen in sents:
                    words = tokenize(sen.rstrip())
                    if reverse:
                        words = words[::-1]
                        punct_ind = None
                        for i, w in enumerate(words):
                            if w in ['.','?','!']:
                                punct_ind = i
                                break
                        punct = words.pop(punct_ind) if punct_ind is not None else ''
                        words = words + [punct]
                    new_sents.append(' '.join(words))
                if split_sentences:
                    for sent in new_sents:
                        out.write(sent + '\n')
                else:
                    out.write(' '.join(new_sents) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file",
        required=True,
        help="text file to process",
    )
    parser.add_argument(
        "--out_file",
        required=True,
        help="output file to write to",
    )
    parser.add_argument(
        "--language",
        help="language code; list of languages can be found at http://www.loc.gov/standards/iso639-2/php/code_list.php",
        default="en"
    )
    parser.add_argument(
        "--reverse",
        action='store_true',
        help="reverse text",
    )
    parser.add_argument(
        "--split_sentences",
        action='store_true',
        help="split documents into individual sentences",
    )
    parser.add_argument(
        "--document_level_processing",
        action='store_true',
        help="perform tokenization and reversal at document level rather than sentence level",
    )
    args = parser.parse_args()
    if args.split_sentences and args.document_level_processing:
        parser.error("Incompatible flags set. Please choose only one of `split_sentences` and `document_level_processing`")
    with open(args.in_file, 'r') as f, open(args.out_file,'w') as o:
        tokenize_strings(f, o, args.language, args.split_sentences, args.reverse, args.document_level_processing)
