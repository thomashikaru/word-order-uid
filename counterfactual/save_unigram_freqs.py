from collections import defaultdict
import pandas as pd
import argparse
from iso_639 import lang_codes
from glob import glob
import os
# from nltk.tokenize import word_tokenize
from mosestokenizer import MosesTokenizer, MosesPunctuationNormalizer

def save_freqs(filename, save_dir, lang):
    d = defaultdict(int)
    with open(filename) as f, MosesTokenizer(lang, no_escape=True) as word_tokenize, MosesPunctuationNormalizer(lang) as normalize:
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
        "--data_dir", help="directory with language data", default="../raw_data/wiki40b"
    )
    parser.add_argument(
        "--save_dir", help="directory to save freq counts", default="freqs"
    )
    args = parser.parse_args()

    langs = args.langs.split(",")
    for lang in langs:
        filename = os.path.join(args.data_dir, f"{lang}.train")
        save_freqs(filename, args.save_dir, lang)
