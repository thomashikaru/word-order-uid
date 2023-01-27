from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import argparse
from tqdm import tqdm
from corpus_iterator_funchead import CorpusIteratorFuncHead
import sys
import json
import pyconll
from scipy.stats import entropy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--language")
    parser.add_argument("--outfile")
    parser.add_argument("--lemmatize", action="store_true")
    args = parser.parse_args()

    subj = defaultdict(int)
    obj = defaultdict(int)
    verb = defaultdict(int)

    subj_lemm = defaultdict(int)
    obj_lemm = defaultdict(int)
    verb_lemm = defaultdict(int)

    corpus = pyconll.load_from_file(args.filename)
    for sentence in corpus:
        for token in sentence:

            if token.deprel.startswith("nsubj"):
                # lemma
                subj_lemm[token.lemma.lower() if token.lemma else token.form] += 1
                verb_lemm[
                    sentence[token.head].lemma.lower()
                    if sentence[token.head].lemma
                    else sentence[token.head].form
                ] += 1

                # non-lemma
                subj[token.form.lower()] += 1
                verb[sentence[token.head].form.lower()] += 1

            if token.deprel.startswith("obj"):
                # lemma
                obj_lemm[token.lemma.lower() if token.lemma else token.form] += 1

                # non-lemma
                obj[token.form.lower()] += 1

    subj_counts = Counter(subj)
    total = sum(subj_counts.values())
    for key in subj_counts:
        subj_counts[key] /= total
    print("Most common subjects:")
    print(subj_counts.most_common(10))

    obj_counts = Counter(obj)
    total = sum(obj_counts.values())
    for key in obj_counts:
        obj_counts[key] /= total
    print("Most common objects:")
    print(obj_counts.most_common(10))

    verb_counts = Counter(verb)
    total = sum(verb_counts.values())
    for key in verb_counts:
        verb_counts[key] /= total
    print("Most common verbs:")
    print(verb_counts.most_common(10))

    subj_counts_lemm = Counter(subj_lemm)
    total = sum(subj_counts.values())
    for key in subj_counts:
        subj_counts[key] /= total
    print("Most common subject lemmas:")
    print(subj_counts.most_common(10))

    obj_counts_lemm = Counter(obj_lemm)
    total = sum(obj_counts.values())
    for key in obj_counts:
        obj_counts[key] /= total
    print("Most common object lemmas:")
    print(obj_counts.most_common(10))

    verb_counts_lemm = Counter(verb_lemm)
    total = sum(verb_counts.values())
    for key in verb_counts:
        verb_counts[key] /= total
    print("Most common verb lemmas:")
    print(verb_counts.most_common(10))

    df = pd.DataFrame(
        {
            "category": ["subject", "object", "verb"],
            "entropy": [
                entropy(list(subj_counts.values())),
                entropy(list(obj_counts.values())),
                entropy(list(verb_counts.values())),
            ],
            "entropy_lemma": [
                entropy(list(subj_counts_lemm.values())),
                entropy(list(obj_counts_lemm.values())),
                entropy(list(verb_counts_lemm.values())),
            ],
            "entropy_top_1k": [
                entropy(list(Counter(dict(subj_counts.most_common(1000))).values())),
                entropy(list(Counter(dict(obj_counts.most_common(1000))).values())),
                entropy(list(Counter(dict(verb_counts.most_common(1000))).values())),
            ],
            "entropy_lemma_top_1k": [
                entropy(
                    list(Counter(dict(subj_counts_lemm.most_common(1000))).values())
                ),
                entropy(
                    list(Counter(dict(obj_counts_lemm.most_common(1000))).values())
                ),
                entropy(
                    list(Counter(dict(verb_counts_lemm.most_common(1000))).values())
                ),
            ],
        }
    )
    df.language = args.language
    df.to_csv(args.outfile, index=False)
