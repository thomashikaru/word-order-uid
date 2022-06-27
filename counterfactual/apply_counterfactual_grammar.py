# Applying a counterfactual grammar to a language dataset
# Original Author: Michael Hahn
# Adapted by: Thomas Hikaru Clark (thclark@mit.edu)

# Usage example (running a grammar optimized for overall efficiency):
# python3 applyCounterfactualGrammar.py \
#   --language English \
#   --model 7580379440 \
#   --base_dir manual_output_funchead_two_coarse_lambda09_best_large \
#   --filename english_sample.conllu

import sys
import random
from collections import deque
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
from corpus_iterator_funchead import CorpusIteratorFuncHead

recursionlimit = sys.getrecursionlimit()
sys.setrecursionlimit(min(4000, 2 * recursionlimit))


def makeCoarse(x):
    """Make coarse, i.e. chop off anything after the colon

    Args:
        x (str): input string

    Returns:
        str: input truncated from colon onwards
    """
    if ":" in x:
        return x[: x.index(":")]
    return x


def initializeOrderTable(filename, language):
    """Get the set of dependency relations used in this particular grammar

    Args:
        filename (str): path to file
        language (str): name of language, e.g. 'English'

    Returns:
        set: set of dependency relations, e.g. {'case', 'nmod'}
    """
    depsVocab = set()
    for sentence, newdoc in CorpusIteratorFuncHead(
        filename, language, validate=False
    ).iterator():
        for line in sentence:
            line["coarse_dep"] = makeCoarse(line["dep"])
            depsVocab.add(line["coarse_dep"])
    return depsVocab


def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
    line = sentence[position - 1]
    allGradients = gradients_from_the_left_sum
    if "children_DH" in line:
        for child in line["children_DH"]:
            allGradients = recursivelyLinearize(sentence, child, result, allGradients)
    result.append(line)
    line["relevant_logprob_sum"] = allGradients
    if "children_HD" in line:
        for child in line["children_HD"]:
            allGradients = recursivelyLinearize(sentence, child, result, allGradients)
    return allGradients


def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax, distanceWeights):
    logits = [
        (x, distanceWeights[sentence[x - 1]["dependency_key"]])
        for x in remainingChildren
    ]
    logits = sorted(logits, key=lambda x: x[1], reverse=(not reverseSoftmax))
    childrenLinearized = list(map(lambda x: x[0], logits))
    return childrenLinearized


def orderSentence(sentence, model, dhWeights, distanceWeights, debug=False):
    root = None

    # for factual ordering, some items will be eliminated (punctuation)
    if model == "REAL_REAL":
        eliminated = []

    # iterate over lines in the parse (i.e. over words in the sentence)
    for line in sentence:

        # make the dependency relation label coarse (ignore stuff after colon)
        line["coarse_dep"] = makeCoarse(line["dep"])

        # identify the root
        if line["coarse_dep"] == "root":
            root = line["index"]
            continue

        # for factual ordering, add punctuation to list of items to be eliminated
        # assumes that punctuation does not have non-punctuation dependents!
        if line["coarse_dep"].startswith("punct"):
            if model == "REAL_REAL":
                eliminated.append(line)
            continue

        # this will be used later
        key = line["coarse_dep"]
        line["dependency_key"] = key

        # do some fancy stuff, not exactly sure what this does
        direction = "DH" if (model == "REAL_REAL" or dhWeights.get(key) > 0) else "HD"
        headIndex = line["head"] - 1
        sentence[headIndex]["children_" + direction] = sentence[headIndex].get(
            "children_" + direction, []
        ) + [line["index"]]

    # for factual ordering, handle eliminations
    if model == "REAL_REAL":
        while len(eliminated) > 0:
            line = eliminated[0]
            del eliminated[0]
            if "removed" in line:
                continue
            line["removed"] = True
            if "children_DH" in line:
                assert 0 not in line["children_DH"]
                eliminated = eliminated + [sentence[x - 1] for x in line["children_DH"]]
            if "children_HD" in line:
                assert 0 not in line["children_HD"]
                eliminated = eliminated + [sentence[x - 1] for x in line["children_HD"]]

        # a sentence is a list of dicts; filter out dicts that were removed
        linearized = filter(lambda x: "removed" not in x, sentence)

    # for other orderings, order children relatively
    else:
        for line in sentence:
            if "children_DH" in line:
                childrenLinearized = orderChildrenRelative(
                    sentence, line["children_DH"][:], False, distanceWeights
                )
                line["children_DH"] = childrenLinearized
            if "children_HD" in line:
                childrenLinearized = orderChildrenRelative(
                    sentence, line["children_HD"][:], True, distanceWeights
                )
                line["children_HD"] = childrenLinearized

        # recursively linearize a sentence
        linearized = []
        recursivelyLinearize(
            sentence, root, linearized, Variable(torch.FloatTensor([0.0]))
        )

        # store new dependency links
        moved = [None] * len(sentence)
        for i, x in enumerate(linearized):
            moved[x["index"] - 1] = i
        for i, x in enumerate(linearized):
            if x["head"] == 0:  # root
                x["reordered_head"] = 0
            else:
                x["reordered_head"] = 1 + moved[x["head"] - 1]

    if debug:
        print("Original")
        print(" ".join(list(map(lambda x: x["word"], sentence))))
        print("Linearized")
        print(" ".join(list(map(lambda x: x["word"], linearized))))

    return linearized


def get_model_specs(filename, model, language, base_dir):
    """Retrieve the model specifications from the grammar descriptions file,
    or generate random grammar specifications if model=='RANDOM'

    Args:
        filename (str): path to dataset file
        model (str): name of model, e.g. 'REAL_REAL', 'RANDOM', or '7580379440'
        language (str): name of language, e.g. 'English'
        base_dir (str): path to directory containing grammar description file

    Returns:
        Tuple[dict, dict]: two dictionaries corresponding to 1) dependency-head weights
        and 2) distance weights
    """

    dhWeights = {}
    distanceWeights = {}

    # Grammar can be one of the following:
    # a) RANDOM grammar
    # b) REAL_REAL, meaning the factual order
    # c) an optimized grammar from a grammar file

    # handle the grammar specification and populate the dhWeights and distanceWeights dicts
    if model.startswith("RANDOM"):  # a random ordering
        # depsVocab = initializeOrderTable(filename, language)
        # itos_deps = sorted(depsVocab)
        # for x in itos_deps:
        #     dhWeights[x] = random.random() - 0.5
        #     distanceWeights[x] = random.random() - 0.5
        grammar_file = os.path.join(base_dir, "auto-summary-lstm.tsv")
        df = pd.read_csv(grammar_file, sep="\t")
        df = df[df["Language"] == language]
        deps = sorted(set(df["CoarseDependency"]))
        for x in deps:
            dhWeights[x] = random.random() - 0.5
            distanceWeights[x] = random.random() - 0.5
    elif model == "REAL_REAL":
        pass
    else:
        grammar_file = os.path.join(base_dir, "auto-summary-lstm.tsv")
        with open(grammar_file, "r") as inFile:
            data = [x.split("\t") for x in inFile.read().strip().split("\n")]
            header = data[0]
            data = data[1:]

        if "CoarseDependency" not in header:
            header[header.index("Dependency")] = "CoarseDependency"
        if "DH_Weight" not in header:
            header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
        if "DistanceWeight" not in header:
            header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

        for line in data:
            if (
                line[header.index("FileName")] == model
                and line[header.index("Language")] == language
            ):
                key = line[header.index("CoarseDependency")]
                dhWeights[key] = float(line[header.index("DH_Weight")])
                distanceWeights[key] = float(line[header.index("DistanceWeight")])

    return dhWeights, distanceWeights


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", help="name of language, e.g. English", default="English"
    )
    parser.add_argument(
        "--model",
        help="type of model, e.g. RANDOM, REAL_REAL, or numerical ID of grammar",
        default="7580379440",
    )
    parser.add_argument(
        "--base_dir",
        help="base directory of grammar file",
        default="../grammars/manual_output_funchead_two_coarse_lambda09_best_large",
    )
    parser.add_argument(
        "--filename", help="filename of CONLLU data", default="en.tiny.conllu"
    )
    parser.add_argument(
        "--seed", help="random seed for making RANDOM grammars", type=int, default=1
    )
    args = parser.parse_args()

    random.seed(args.seed)
    dhWeights, distanceWeights = get_model_specs(
        args.filename, args.model, args.language, args.base_dir
    )

    # load and iterate over a corpus
    corpus = CorpusIteratorFuncHead(args.filename, args.language, "train")
    corpusIterator = corpus.iterator()
    for i, (sentence, newdoc) in enumerate(corpusIterator):
        ordered = orderSentence(sentence, args.model, dhWeights, distanceWeights)
        output = " ".join(list(map(lambda x: x["word"], ordered)))

        # Add a new line if the just-processed sentence starts a new document
        if newdoc and i != 0:
            sys.stdout.write("\n")

        sys.stdout.write(output)
        sys.stdout.write(" . ")  # add a period after every sentence

