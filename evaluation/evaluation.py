# Python helper script for post-processing the raw token-by-token surprisals
#   resulting from the eval-lm stage
# Assumes there are .pt files containing tokens and surprisals for the
#   various combinations of language and model
# Run with flag --make_csv to saves a CSV containing all surprisals
# Run with flag --plot_dl_v_surp to make a dl vs surp plot


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import glob
import string
import torch
import re
from scipy.stats import zscore
import argparse

# mapping from numerical grammar IDs (see grammars in counterfactual/grammars)
# to human-readable descriptors
mapping = {
    "1654461679": "Efficient-OV",
    "9864186953": "Efficient-OV",
    "4910096554": "Efficient-OV",
    "7233494255": "Efficient-OV",
    "1804798267": "Efficient-OV",
    "1035393965": "Efficient-OV",
    "5786187046": "Efficient-OV",
    "3375856929": "Efficient-OV",
    "5457228368": "Efficient-VO",
    "1171173532": "Efficient-VO",
    "6448967977": "Efficient-VO",
    "3150957569": "Efficient-VO",
    "6912153951": "Efficient-VO",
    "8151228474": "Efficient-VO",
    "9615205925": "Efficient-VO",
    "4418369424": "Efficient-VO",
    "1935936": "Approx",
    "1988915": "Approx",
    "7520227": "Approx",
    "6522123": "Approx",
    "9269015": "Approx",
    "5754928": "Approx",
    "1920622": "Approx",
    "3564332": "Approx",
    "8850185": "Min-DL-Opt",
    "514370": "Min-DL-Opt",
    "5940573": "Min-DL-Opt",
    "2103598": "Min-DL-Opt",
    "2684503": "Min-DL-Opt",
    "1355012": "Min-DL-Opt",
    "5333556": "Min-DL-Opt",
    "6786312": "Min-DL-Opt",
    "REAL_REAL": "Real",
    "RANDOM-1": "Random-1",
    "RANDOM-2": "Random-2",
    "SORT_FREQ": "Sort-Freq",
    "SORT_FREQ_REV": "Sort-Freq-Rev",
    "REVERSE": "Reverse",
    "MIN_DL_PROJ": "Min-DL-Loc",
}


def aggregate_bpe(a, b):
    """Takes a list of tokens and a list of surprisal values and 
    aggregates surprisals across multiple BPE tokens, returning a new
    list of aggregated tokens and a new list of aggregated surprisals.

    Args:
        a (list[str]): list of tokens
        b (list[float]): list of surprisals, b[i] corresponds to token a[i]

    Returns:
        tuple[list[str], list[float]]: new lists of tokens and surprisals
    """

    assert len(a) == len(b), "Expected length of tokens and surprisals to match"

    tokens, surprisals = [], []

    for key, grp in itertools.groupby(zip(a, b), lambda x: "@@" in x[0]):
        toks, surps = zip(*list(grp))
        if key:
            tokens.append("".join(toks).replace("@@", ""))
            surprisals.append(sum(surps))
        elif len(tokens) > 0:
            tokens[-1] += toks[0]
            surprisals[-1] += surps[0]
            tokens += toks[1:]
            surprisals += surps[1:]
        else:
            tokens += toks
            surprisals += surps

    return tokens, surprisals


def aggregate_sentencepiece(a, b):
    """Takes a list of tokens and a list of surprisal values and 
    aggregates surprisals across multiple sentencepiece tokens, returning a new
    list of aggregated tokens and a new list of aggregated surprisals.

    Args:
        a (list[str]): list of tokens
        b (list[float]): list of surprisals, b[i] corresponds to token a[i]

    Returns:
        tuple[list[str], list[float]]: new lists of tokens and surprisals
    """

    assert len(a) == len(b), "Expected length of tokens and surprisals to match"

    tokens, surprisals = [], []

    for key, grp in itertools.groupby(zip(a, b), lambda x: x[0].startswith("▁")):
        toks, surps = zip(*list(grp))
        if key:
            tokens += [tok.replace("▁", "") for tok in toks]
            surprisals += surps
        elif len(tokens) > 0:
            tokens[-1] += "".join(toks)
            surprisals[-1] += sum(surps)
        else:
            raise Exception("this shouldn't happen")

    return tokens, surprisals


def doc_stats(df):
    """Adds the columns 'sentence_id' and 'sentence_pos' to a df

    Args:
        df (pd.DataFrame): input dataframe containing one row for each
        word in the evaluation

    Returns:
        pd.DataFrame: new dataframe with 2 new columns added
    """
    sentence_pos = []
    sentence_ids = []
    counter = 0
    sentence_check = lambda x: x == "."

    # group by presence of period (.), used as EOS token
    for k, g in itertools.groupby(df["token"], sentence_check):
        g_list = list(g)
        if not k:
            sentence_ids.extend([counter] * (len(g_list)))
            sentence_pos.extend(list(range(len(g_list))))
        else:
            if counter == 0:
                sentence_ids.extend([0] * len(g_list))
            else:
                sentence_ids.extend([counter] * len(g_list))

            if len(sentence_pos) == 0:
                sentence_pos.extend(list(range(0, len(g_list))))
            else:
                val = sentence_pos[-1] + 1
                sentence_pos.extend(list(range(val, val + len(g_list))))

            counter += 1

    df["sentence_id"] = sentence_ids
    df.sentence_id = df.sentence_id.astype("int16")
    df["sentence_pos"] = sentence_pos
    df.sentence_pos = df.sentence_pos.astype("int16")

    return df


def make_df(
    surprisals_list, tokens_list, lang, variant, agg_bpe=True, agg_sentencepiece=False
):
    """Make a dataframe from a list of surprisals and a list of tokens

    Args:
        surprisals_list (list[float]): list of surprisals
        tokens_list (list[str]): list of tokens
        lang (str): language name
        variant (str): grammar variant name
        agg_bpe (bool, optional): set to true to aggregate BPE tokens additively. Defaults to True.
        agg_sentencepiece (bool, optional): set to true to agggregate sentencepiece tokens additively. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containing the data with informative columns added
    """
    positions = []
    final_surprisals = []
    final_tokens = []
    document_lens = []
    document_ids = []
    for j, (surprisals, tokens) in enumerate(zip(surprisals_list, tokens_list)):

        if agg_bpe:
            tokens, surprisals = aggregate_bpe(tokens, surprisals)

        if agg_sentencepiece:
            tokens, surprisals = aggregate_sentencepiece(tokens, surprisals)

        assert len(tokens) == len(surprisals)

        doc_len = len(tokens)
        for i, surprisal in enumerate(surprisals):
            positions.append(i)
            final_surprisals.append(surprisal)
            final_tokens.append(tokens[i])
            document_lens.append(doc_len)
            document_ids.append(j)

    df = pd.DataFrame(
        {
            "doc_pos": positions,
            "surprisal": final_surprisals,
            "token": final_tokens,
            "document_len": document_lens,
            "document_id": document_ids,
        }
    )

    df["language"] = lang
    df.language = df.language.astype("category")
    df["variant"] = variant
    df.variant = df.variant.astype("category")

    df.doc_pos = df.doc_pos.astype("int16")
    df.surprisal = df.surprisal.astype("float32")
    df.document_len = df.document_len.astype("int16")
    df.document_id = df.document_id.astype("int16")

    df = df.groupby("document_id").apply(doc_stats).reset_index(drop=True)

    return df


def remove_trailing(df):
    """Due to batch size cutoff, a document may get cut off in the middle of a sentence. This
    function removes any material occuring after the final period in a document.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: new dataframe with trailing material removed
    """
    max_sent_id = max(df.sentence_id)

    if df.iloc[-1].token != ".":
        return df[df.sentence_id != max_sent_id]

    return df


def make_csv(args):
    """Read in the evaluation data and make a giant dataframe from all of it. Save to file.

    Returns:
        pd.DataFrame: dataframe
    """
    dfs = []
    files = glob.glob(args.perps_file_pattern)

    # read each file, get surprisal and token lists, and feed to make_df()
    for file in files:
        name = file.split("/")[-1].split(".")[0]
        lang = name.split("-")[0]
        model = "-".join(name.split("-")[1:])
        dat = torch.load(file)
        logprob_lists = map(lambda x: x.numpy(), dat[0])
        token_lists = dat[1]

        dfs.append(
            make_df(
                logprob_lists,
                token_lists,
                lang=lang,
                variant=model,
                agg_bpe=True,
                agg_sentencepiece=False,
            )
        )
    df = pd.concat(dfs, ignore_index=True)

    # apply the remove_trailing() function to each document
    # (see the function docstring for explanation)
    df = (
        df.groupby(["language", "variant", "document_id"])
        .apply(remove_trailing)
        .reset_index(drop=True)
    )

    # save to file
    df = df.apply(lambda x: np.negative(x) if x.name == "surprisal" else x)
    df["variant"] = df["variant"].apply(lambda x: mapping.get(x, x))
    df.variant = df.variant.astype("category")
    df.language = df.language.astype("category")
    df.to_feather(args.out_file)
    return df


def plot_dl_vs_surp():
    """This is part of a secondary analysis that was not included in the paper.
    Briefly, it looks at the relationship between average dependency length and
    surprisal, but only does this for some languages currently. 
    """
    files = glob.glob("../counterfactual/dep-len/*-test.txt")
    d = []
    for file in files:

        basename = file.split("/")[-1].split(".")[0]
        pieces = basename.split("-")
        lang = pieces[0]
        model = "-".join(pieces[1:-1])
        part = pieces[-1]

        with open(file) as f:
            data = f.readlines()[-1]
            dl = float(data.split(": ")[-1])

            d.append({"lang": lang, "model": model, "part": part, "dl": dl})

    df = pd.DataFrame(d)

    df = df[df.part == "test"]

    df["model"] = df["model"].replace(mapping)
    df["dataset"] = df["lang"] + "-" + df["model"]

    df.to_csv("dep-len-by-dataset.csv", index=False)

    order = [
        "RANDOM-1",
        "RANDOM-2",
        "OV",
        "VO",
        "REVERSE",
        "Approx",
        "REAL",
        "MIN_DL_PROJ",
        "MIN_DL_OPT",
        "SORT_FREQ",
    ]

    colors = [
        "khaki",
        "gold",
        "greenyellow",
        "lightgreen",
        "red",
        "deepskyblue",
        "royalblue",
        "darkorchid",
        "mediumorchid",
        "plum",
    ]

    pal = dict(zip(order, colors))

    g = sns.catplot(
        data=df,
        x="model",
        y="dl",
        hue="model",
        col="lang",
        col_wrap=4,
        kind="bar",
        order=order,
        hue_order=order,
        palette=pal,
        dodge=False,
    )
    g.set_xticklabels(rotation=90)
    g.add_legend(fontsize=16, loc="upper right")
    plt.savefig("dep-lens-by-dataset", dpi=150, bbox_inches="tight")

    surps = pd.read_csv("eval_results_cf_v4/cf_eval_data_8langs.csv")
    surps = (
        surps.groupby(["language", "variant"]).agg({"surprisal": np.mean}).reset_index()
    )
    surps["dataset"] = surps["language"] + "-" + surps["variant"]
    surps["dl"] = surps["dataset"].replace(dict(zip(df.dataset, df.dl)))
    surps["surp_z"] = surps["surprisal"].groupby(surps["language"]).transform(zscore)
    surps["dl_z"] = surps["dl"].groupby(surps["language"]).transform(zscore)

    surps.to_csv("dl_v_surp_v4.csv", index=False)

    fig, axes = plt.subplots(figsize=(16, 8))
    sns.regplot(data=surps, x="dl_z", y="surp_z")
    plt.savefig("surpz-vs-dlz-reg", dpi=150, bbox_inches="tight")

    px.defaults.width = 800
    px.defaults.height = 600
    p = px.scatter(
        surps,
        x="dl_z",
        y="surp_z",
        hover_data=["dataset"],
        color="language",
        trendline="ols",
    )
    p.write_image("surpz-vs-dlz.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--make_csv", action="store_true")
    parser.add_argument("--plot_dl_vs_surp", action="store_true")
    parser.add_argument("--perps_file_pattern")
    parser.add_argument("--out_file")
    args = parser.parse_args()

    if args.make_csv:
        df = make_csv()

    if args.plot_dl_vs_surp:
        sns.set_style("dark")
        sns.set(font_scale=2)
        plot_dl_vs_surp()

