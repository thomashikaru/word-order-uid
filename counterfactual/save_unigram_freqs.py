from collections import defaultdict
import pandas as pd
import argparse
from iso_639 import lang_codes
from glob import glob
import os

# from nltk.tokenize import word_tokenize
from mosestokenizer import MosesTokenizer, MosesPunctuationNormalizer
from indicnlp.tokenize.indic_tokenize import trivial_tokenize as indic_word_tokenize
from hazm import word_tokenize as persian_word_tokenize


def save_freqs(filename, save_dir, lang):
    d = defaultdict(int)

    if lang == "fa":
        word_tokenize = persian_word_tokenize
    if lang == "hi":
        word_tokenize = indic_word_tokenize
    else:
        word_tokenize = MosesTokenizer(lang, no_escape=True)

    with open(filename) as f:
        for line in f:
            for word in word_tokenize(line):
                d[word] += 1

    df = pd.DataFrame.from_records(iter(d.items()), columns=["word", "count"])
    count_sum = df["count"].sum()
    df["freq"] = (df["count"] / count_sum).astype("float32")
    df.sort_values("count", ascending=False, inplace=True)
    df.to_csv(f"{save_dir}/{lang}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langs",
        help="comma-separated list of languages codes for which to get unigram freqs",
        default="en,ru,fr,de,vi,id,hu,tr",
    )
    parser.add_argument(
        "--data_dir",
        help="directory with language data",
        default="../data/raw_data/wiki40b-txt-sampled",
    )
    parser.add_argument(
        "--save_dir", help="directory to save freq counts", default="freqs"
    )
    args = parser.parse_args()

    langs = args.langs.split(",")
    for lang in langs:
        filename = os.path.join(args.data_dir, f"{lang}.train")
        save_freqs(filename, args.save_dir, lang)
