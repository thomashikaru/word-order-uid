import sys
import argparse

from mosestokenizer import MosesTokenizer, MosesSentenceSplitter
from string import punctuation

PUNCT_LIST = [".", "?", "!"]


def reverse_tokens(words):
    words = words[::-1]
    punct_ind = None
    # for i, w in enumerate(words):
    #     if w in PUNCT_LIST:
    #         punct_ind = i
    #         break
    if words[0] in PUNCT_LIST:
        punct_ind = 0
    punct = words.pop(punct_ind) if punct_ind is not None else ""
    words = words + [punct]
    return words


def tokenize_strings_seq2seq(f, out1, out2, args):
    with MosesTokenizer(args.language) as tokenize, MosesSentenceSplitter(
        args.language
    ) as splitsents:
        for line in f:
            if line.isspace():
                out1.write("\n")
                out2.write("\n")
                continue
            sents = splitsents([line])
            context = []
            for i, sen in enumerate(sents):
                words = tokenize(sen.rstrip())

                words[0] = words[0].lower()
                words_out = reverse_tokens(words) if args.reverse else words

                if i != 0:
                    out1.write(" ".join(context) + "\n")
                    out2.write(" ".join(words_out) + "\n")
                context.append(" ".join(words))


def tokenize_strings(f, out, args):
    with MosesTokenizer(args.language) as tokenize, MosesSentenceSplitter(
        args.language
    ) as splitsents:
        for line in f:
            if line.isspace():
                out.write("\n")
                continue
            if args.document_level_processing:
                words = tokenize(line.rstrip())
                words = words[::-1] if args.reverse else words
                out.write(" ".join(words) + "\n")
            elif args.seq2seq:
                sents = splitsents([line])
                new_sents = []
                for i, sen in enumerate(sents):
                    words = tokenize(sen.rstrip())
            else:
                sents = splitsents([line])
                new_sents = []
                for sen in sents:
                    words = tokenize(sen.rstrip())
                    if args.reverse:
                        words = reverse_tokens(words)
                    new_sents.append(" ".join(words))
                if args.split_sentences:
                    for sent in new_sents:
                        out.write(sent + "\n")
                else:
                    out.write(" ".join(new_sents) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file", required=True, help="text file to process",
    )
    parser.add_argument(
        "--out_file", required=True, help="output file to write to",
    )
    parser.add_argument(
        "--out_file2", help="second output file to write to, for seq2seq",
    )
    parser.add_argument(
        "--language",
        help="language code; list of languages can be found at http://www.loc.gov/standards/iso639-2/php/code_list.php",
        default="en",
    )
    parser.add_argument(
        "--reverse", action="store_true", help="reverse text",
    )
    parser.add_argument(
        "--split_sentences",
        action="store_true",
        help="split documents into individual sentences",
    )
    parser.add_argument(
        "--document_level_processing",
        action="store_true",
        help="perform tokenization and reversal at document level rather than sentence level",
    )
    parser.add_argument(
        "--seq2seq", action="store_true", help="perform seq2seq preprocessing"
    )

    args = parser.parse_args()
    if args.split_sentences and args.document_level_processing:
        parser.error(
            "Incompatible flags set. Please choose only one of `split_sentences` and `document_level_processing`"
        )

    if args.seq2seq:
        with open(args.in_file, "r") as f, open(args.out_file, "w") as o1, open(
            args.out_file2, "w"
        ) as o2:
            tokenize_strings_seq2seq(f, o1, o2, args)
    else:
        with open(args.in_file, "r") as f, open(args.out_file, "w") as o:
            tokenize_strings(f, o, args)
