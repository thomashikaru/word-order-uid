import pandas as pd
import numpy as np
import spacy
import neuralcoref
import argparse
import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="../data/wiki40b-txt/en.tiny")
    args = parser.parse_args()

    nlp = spacy.load("en")
    neuralcoref.add_to_pipe(nlp)

    filenames = iter(glob.glob(args.pattern))

    with open(next(filenames)) as f:
        # for line in f:
        lines = f.readlines()
        line = lines[1]
        print(line)
        doc1 = nlp(line)
        clusters = doc1._.coref_clusters
        for cluster in clusters:
            print(cluster.main, cluster.main.start, cluster.main.end)
            for mention in cluster.mentions:
                print("\t", mention, mention.start, mention.end)
