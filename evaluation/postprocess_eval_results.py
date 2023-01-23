import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import itertools
from variant_name2id import name2id
import glob
import torch
from collections import defaultdict
from scipy.stats import sem

conversion = {
    "APPROX": "Approx",
    "MIN_DL_OPT": "Min-DL-Opt",
    "EFFICIENT_VO": "Efficient-VO",
    "EFFICIENT_OV": "Efficient-OV",
}

mapping = {}

for lang, dic in name2id.items():
    for variant, id in dic.items():
        assert variant in conversion
        mapping[id] = conversion[variant]

mapping.update(
    {
        "REAL_REAL": "Real",
        "RANDOM_1": "Random-1",
        "RANDOM_2": "Random-2",
        "RANDOM_3": "Random-3",
        "RANDOM_4": "Random-4",
        "RANDOM_5": "Random-5",
        "SORT_FREQ": "Sort-Freq",
        "SORT_FREQ_REV": "Sort-Freq-Rev",
        "REVERSE": "Reverse",
        "MIN_DL_PROJ": "Min-DL-Loc",
    }
)


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
    surprisals_list,
    tokens_list,
    lang,
    variant,
    num_toks,
    model_seed,
    agg_bpe=True,
    agg_sentencepiece=False,
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

    if num_toks is not None and model_seed is not None:
        df["num_toks"] = num_toks
        df.num_toks = df.num_toks.astype("category")
        df["model_seed"] = model_seed
        df.model_seed = df.model_seed.astype("category")

    df.doc_pos = df.doc_pos.astype("int16")
    df.surprisal = df.surprisal.astype("float32")
    df.document_len = df.document_len.astype("int16")
    df.document_id = df.document_id.astype("int16")

    df = df.groupby("document_id").apply(doc_stats).reset_index(drop=True)
    df = df.groupby("document_id").apply(remove_trailing).reset_index(drop=True)

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


def postprocess(args):
    """Read in the evaluation data and make a giant dataframe from all of it. Save to file.

    Returns:
        pd.DataFrame: dataframe
    """
    df_summary = defaultdict(list)
    file = args.inputfile

    dat = torch.load(file)
    logprob_lists = map(lambda x: x.numpy(), dat[0])
    token_lists = dat[1]

    df = make_df(
        logprob_lists,
        token_lists,
        lang=lang,
        variant=args.variant,
        num_toks=args.num_toks,
        model_seed=args.model_seed,
        agg_bpe=True,
        agg_sentencepiece=False,
    )

    print(df.info())

    # save to file
    df.surprisal = np.negative(df.surprisal)
    df["variant"] = df["variant"].apply(lambda x: mapping.get(x, x))
    df.variant = df.variant.astype("category")
    df.language = df.language.astype("category")

    if "num_toks" in df.columns and "model_seed" in df.columns:
        df.num_toks = df.num_toks.astype("category")
        df.model_seed = df.model_seed.astype("category")
    print(df.info())

    # df.to_feather(args.out_file)

    df["sentence_len"] = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .sentence_id.transform("count")
        .astype("int16")
    )

    # surprisals
    # df = (
    #     df.groupby(["language", "variant"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/surprisal.csv", index=False)
    df_summary["surprisal_mean"].append(np.mean(df.surprisal))
    df_summary["surprisal_sem"].append(sem(df.surprisal))
    df_summary["surprisal_std"].append(np.std(df.surprisal))

    # Surprisal Variance
    # df = (
    #     df.groupby(
    #         ["language", "variant", "document_id", "sentence_id"], observed=True
    #     )
    #     .surprisal.agg(np.var)
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/surprisal_variance.csv", index=False)
    d = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .surprisal.agg(np.var)
        .dropna()
        .reset_index()
    )
    df_summary["surprisal_var_mean"].append(np.mean(d.surprisal))
    df_summary["surprisal_var_sem"].append(sem(d.surprisal))
    df_summary["surprisal_var_std"].append(np.std(d.surprisal))

    # Document-initial surprisal variance
    # df = (
    #     df.query("sentence_id == 0")
    #     .copy()
    #     .groupby(
    #         ["language", "variant", "document_id", "sentence_id"], observed=True
    #     )
    #     .surprisal.agg(np.var)
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/doc_initial_var.csv", index=False)
    d = (
        df.query("sentence_id == 0")
        .groupby(["document_id", "sentence_id"], observed=True)
        .surprisal.agg(np.var)
        .dropna()
        .reset_index()
    )
    df_summary["surprisal_var_doc_initial_mean"].append(np.mean(d.surprisal))
    df_summary["surprisal_var_doc_initial_sem"].append(sem(d.surprisal))
    df_summary["surprisal_var_doc_initial_std"].append(np.std(d.surprisal))

    # Surprisal Deviation from Dataset mean
    # df["surp_diff_squared"] = (
    #     df.surprisal
    #     - df.groupby(["language", "variant"], observed=True).surprisal.transform(
    #         "mean"
    #     )
    # ) ** 2
    # df = (
    #     df.groupby(
    #         ["language", "variant", "document_id", "sentence_id"], observed=True
    #     )
    #     .surp_diff_squared.agg("mean")
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"surp_diff_squared": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/surprisal_deviations.csv", index=False)
    df["surp_diff_squared"] = (df.surprisal - df.surprisal.mean()) ** 2
    d = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .surp_diff_squared.agg(np.mean)
        .dropna()
        .reset_index()
    )
    df_summary["surp_diff_squared_mean"].append(np.mean(d.surp_diff_squared))
    df_summary["surp_diff_squared_sem"].append(sem(d.surp_diff_squared))
    df_summary["surp_diff_squared_std"].append(np.std(d.surp_diff_squared))

    # Avg token-to-token change in surprisal
    # df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
    # mask = (df.sentence_pos == 0) & (df.sentence_id == 0)
    # df.delta_surp[mask] = np.nan
    # df = (
    #     df.dropna(subset=["delta_surp"])
    #     .groupby(["language", "variant", "document_id", "sentence_id"], observed=True)
    #     .agg({"delta_surp": np.mean})
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"delta_surp": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/delta_surps.csv", index=False)
    df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
    mask = (df.sentence_pos == 0) & (df.sentence_id == 0)
    df.delta_surp[mask] = np.nan
    d = (
        df.dropna(subset=["delta_surp"])
        .groupby(["document_id", "sentence_id"], observed=True)
        .delta_surp.agg(np.mean)
        .dropna()
        .reset_index()
    )
    df_summary["delta_surp_mean"].append(np.mean(d.delta_surp))
    df_summary["delta_surp_sem"].append(sem(d.delta_surp))
    df_summary["delta_surp_std"].append(np.std(d.delta_surp))

    # UID Power (k=1.1)
    # k = 1.1
    # df = (
    #     df.groupby(["language", "variant", "document_id", "sentence_id"], observed=True)
    #     .agg({"surprisal": lambda x: ((x ** k).sum()) ** (1 / k)})
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/infs_1.1.csv", index=False)
    k = 1.1
    d = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .surprisal.agg(lambda x: ((x ** k).sum()) ** (1 / k))
        .dropna()
        .reset_index()
    )
    df_summary["uidp_1_1_mean"].append(np.mean(d.surprisal))
    df_summary["uidp_1_1_sem"].append(sem(d.surprisal))
    df_summary["uidp_1_1_std"].append(np.std(d.surprisal))

    # UID Power (k=1.25)
    # k = 1.25
    # df = (
    #     df.groupby(["language", "variant", "document_id", "sentence_id"], observed=True)
    #     .agg({"surprisal": lambda x: ((x ** k).sum()) ** (1 / k)})
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/infs_1.25.csv", index=False)
    k = 1.25
    d = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .surprisal.agg(lambda x: ((x ** k).sum()) ** (1 / k))
        .dropna()
        .reset_index()
    )
    df_summary["uidp_1_25_mean"].append(np.mean(d.surprisal))
    df_summary["uidp_1_25_sem"].append(sem(d.surprisal))
    df_summary["uidp_1_25_std"].append(np.std(d.surprisal))

    # Max surprisal
    # df = (
    #     df.groupby(["language", "variant", "document_id", "sentence_id"], observed=True)
    #     .surprisal.agg("max")
    #     .dropna()
    #     .reset_index()
    #     .groupby(["language", "variant"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/max_surps.csv", index=False)
    d = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .surprisal.agg(np.max)
        .dropna()
        .reset_index()
    )
    df_summary["max_surp_mean"].append(np.mean(d.surprisal))
    df_summary["max_surp_sem"].append(sem(d.surprisal))
    df_summary["max_surp_std"].append(np.std(d.surprisal))

    # surprisal variance (divisive normalization)
    df["surprisal_norm"] = df.surprisal / df_summary["surprisal_mean"][0]
    d = (
        df.groupby(["document_id", "sentence_id"], observed=True)
        .surprisal_norm.agg(np.var)
        .dropna()
        .reset_index()
    )
    df_summary["surprisal_var_norm_mean"].append(np.mean(d.surprisal_norm))
    df_summary["surprisal_var_norm_sem"].append(sem(d.surprisal_norm))
    df_summary["surprisal_var_norm_std"].append(np.std(d.surprisal_norm))

    df_summary = pd.DataFrame(df_summary)
    df_summary["language"] = args.language
    df_summary["variant"] = args.variant
    df_summary["num_toks"] = args.num_toks
    df_summary["model_seed"] = args.model_seed
    df_summary["dataset"] = args.dataset

    # Avg surprisal by token pos
    # df = (
    #     df.query("sentence_len <= 20 & sentence_len >= 10")
    #     .copy()
    #     .groupby(["language", "variant", "sentence_len", "sentence_pos"], observed=True)
    #     .agg({"surprisal": [np.mean, sem, np.std]})
    #     .dropna()
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # df.to_csv(f"{args.data_dir}/avg_surps.csv", index=False)

    # Delta surprisal by token pos
    # df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
    # df.delta_surp.loc[(df.sentence_pos == 0) & (df.sentence_id == 0)] = np.nan
    # df = (
    #     df.dropna(subset=["delta_surp"])
    #     .groupby(["language", "variant", "sentence_len", "sentence_pos"], observed=True)
    #     .agg({"delta_surp": [np.mean, sem, np.std]})
    #     .dropna()
    #     .reset_index()
    # )
    # df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
    # (
    #     df.query("sentence_len >= 10 & sentence_len <= 20").to_csv(
    #         f"{args.data_dir}/delta_surps_by_tok.csv", index=False
    #     )
    # )

    df_summary.to_csv(args.inputfile.replace(".pt", ".csv"), index=False)

    # return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    parser.add_argument("--language")
    parser.add_argument("--variant")
    parser.add_argument("--num_toks")
    parser.add_argument("--model_seed")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    postprocess(args)

