import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from scipy.stats import sem


def main(args):

    # load dataframe
    df = pd.read_feather(args.inputfile)

    df.language = df.language.astype("category")
    df.variant = df.variant.astype("category")

    df["sentence_len"] = (
        df.groupby(["language", "variant", "document_id", "sentence_id"], observed=True)
        .sentence_id.transform("count")
        .astype("int16")
    )

    if args.metric_name == "surprisal":

        # surprisals
        df = (
            df.groupby(["language", "variant"], observed=True)
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/surprisal.csv", index=False)

    if args.metric_name == "surprisal_variance":

        # Surprisal Variance
        df = (
            df.groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .surprisal.agg(np.var)
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/surprisal_variance.csv", index=False)

    if args.metric_name == "doc_initial_var":

        # Document-initial surprisal variance
        df = (
            df.query("sentence_id == 0")
            .copy()
            .groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .surprisal.agg(np.var)
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/doc_initial_var.csv", index=False)

    if args.metric_name == "surprisal_deviations":

        # Surprisal Deviation from Dataset mean
        df["surp_diff_squared"] = (
            df.surprisal
            - df.groupby(["language", "variant"], observed=True).surprisal.transform(
                "mean"
            )
        ) ** 2
        df = (
            df.groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .surp_diff_squared.agg("mean")
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"surp_diff_squared": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/surprisal_deviations.csv", index=False)

    if args.metric_name == "delta_surps":

        # Avg token-to-token change in surprisal
        df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
        mask = (df.sentence_pos == 0) & (df.sentence_id == 0)
        df.delta_surp[mask] = np.nan
        df = (
            df.dropna(subset=["delta_surp"])
            .groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .agg({"delta_surp": np.mean})
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"delta_surp": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/delta_surps.csv", index=False)

    if args.metric_name == "infs_1.1":

        # UID Power (k=1.1)
        k = 1.1
        df = (
            df.groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .agg({"surprisal": lambda x: ((x ** k).sum()) ** (1 / k)})
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/infs_1.1.csv", index=False)

    if args.metric_name == "infs_1.25":

        # UID Power (k=1.25)
        k = 1.25
        df = (
            df.groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .agg({"surprisal": lambda x: ((x ** k).sum()) ** (1 / k)})
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/infs_1.25.csv", index=False)

    if args.metric_name == "avg_surps":

        # Avg surprisal by token pos
        df = (
            df.query("sentence_len <= 20 & sentence_len >= 10")
            .copy()
            .groupby(
                ["language", "variant", "sentence_len", "sentence_pos"], observed=True
            )
            .agg({"surprisal": [np.mean, sem, np.std]})
            .dropna()
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/avg_surps.csv", index=False)

    if args.metric_name == "delta_surps_by_tok":

        # Delta surprisal by token pos
        df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
        df.delta_surp.loc[(df.sentence_pos == 0) & (df.sentence_id == 0)] = np.nan
        df = (
            df.dropna(subset=["delta_surp"])
            .groupby(
                ["language", "variant", "sentence_len", "sentence_pos"], observed=True
            )
            .agg({"delta_surp": [np.mean, sem, np.std]})
            .dropna()
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        (
            df.query("sentence_len >= 10 & sentence_len <= 20").to_csv(
                f"{args.data_dir}/delta_surps_by_tok.csv", index=False
            )
        )

    if args.metric_name == "max_surps":

        # Max surprisal
        df = (
            df.groupby(
                ["language", "variant", "document_id", "sentence_id"], observed=True
            )
            .surprisal.agg("max")
            .dropna()
            .reset_index()
            .groupby(["language", "variant"], observed=True)
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        df.columns = df.columns.get_level_values(0) + df.columns.get_level_values(1)
        df.to_csv(f"{args.data_dir}/max_surps.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    parser.add_argument("--data_dir")
    parser.add_argument("--metric_name")
    args = parser.parse_args()

    main(args)
