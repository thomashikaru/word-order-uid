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
    df["sentence_len"] = (
        df.groupby(["language", "variant", "document_id", "sentence_id"])
        .sentence_id.transform("count")
        .astype("int16")
    )

    if args.metric_name == "surprisal":

        # surprisals
        surprisals = (
            df.groupby(["language", "variant"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        surprisals.columns = surprisals.columns.get_level_values(
            0
        ) + surprisals.columns.get_level_values(1)
        surprisals.to_csv(f"{args.data_dir}/surprisal.csv", index=False)

    if args.metric_name == "surprisal_variance":

        # Surprisal Variance
        surprisal_variances = (
            df.groupby(["language", "variant", "document_id", "sentence_id"])
            .surprisal.agg(np.var)
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        surprisal_variances.columns = surprisal_variances.columns.get_level_values(
            0
        ) + surprisal_variances.columns.get_level_values(1)
        surprisal_variances.to_csv(
            f"{args.data_dir}/surprisal_variance.csv", index=False
        )

    if args.metric_name == "doc_initial_var":

        # Document-initial surprisal variance
        doc_initial_var = (
            df.query("sentence_id == 0")
            .copy()
            .groupby(["language", "variant", "document_id", "sentence_id"])
            .surprisal.agg(np.var)
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        doc_initial_var.columns = doc_initial_var.columns.get_level_values(
            0
        ) + doc_initial_var.columns.get_level_values(1)
        doc_initial_var.to_csv(f"{args.data_dir}/doc_initial_var.csv", index=False)

    if args.metric_name == "surprisal_deviations":

        # Surprisal Deviation from Dataset mean
        df["surp_diff_squared"] = (
            df.surprisal
            - df.groupby(["language", "variant"]).surprisal.transform("mean")
        ) ** 2
        surprisal_deviations = (
            df.groupby(["language", "variant", "document_id", "sentence_id"])
            .surp_diff_squared.agg("mean")
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"surp_diff_squared": [np.mean, sem, np.std]})
            .reset_index()
        )
        surprisal_deviations.columns = surprisal_deviations.columns.get_level_values(
            0
        ) + surprisal_deviations.columns.get_level_values(1)
        surprisal_deviations.to_csv(
            f"{args.data_dir}/surprisal_deviations.csv", index=False
        )

    if args.metric_name == "delta_surps":

        # Avg token-to-token change in surprisal
        df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
        mask = (df.sentence_pos == 0) & (df.sentence_id == 0)
        df.delta_surp[mask] = np.nan
        delta_surps = (
            df.dropna(subset=["delta_surp"])
            .groupby(["language", "variant", "document_id", "sentence_id"])
            .agg({"delta_surp": np.mean})
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"delta_surp": [np.mean, sem, np.std]})
            .reset_index()
        )
        delta_surps.columns = delta_surps.columns.get_level_values(
            0
        ) + delta_surps.columns.get_level_values(1)
        delta_surps.to_csv(f"{args.data_dir}/delta_surps.csv", index=False)

    if args.metric_name == "infs_1.1":

        # UID Power (k=1.1)
        k = 1.1
        infs_1_1 = (
            df.groupby(["language", "variant", "document_id", "sentence_id"])
            .agg({"surprisal": lambda x: ((x ** k).sum()) ** (1 / k)})
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        infs_1_1.columns = infs_1_1.columns.get_level_values(
            0
        ) + infs_1_1.columns.get_level_values(1)
        infs_1_1.to_csv(f"{args.data_dir}/infs_1.1.csv", index=False)

    if args.metric_name == "infs_1.25":

        # UID Power (k=1.25)
        k = 1.25
        infs = (
            df.groupby(["language", "variant", "document_id", "sentence_id"])
            .agg({"surprisal": lambda x: ((x ** k).sum()) ** (1 / k)})
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        infs.columns = infs.columns.get_level_values(0) + infs.columns.get_level_values(
            1
        )
        infs.to_csv(f"{args.data_dir}/infs_1.25.csv", index=False)

    if args.metric_name == "avg_surps":

        # Avg surprisal by token pos
        avg_surps = (
            df.query("sentence_len <= 20 & sentence_len >= 10")
            .copy()
            .groupby(["language", "variant", "sentence_len", "sentence_pos"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .dropna()
            .reset_index()
        )
        avg_surps.columns = avg_surps.columns.get_level_values(
            0
        ) + avg_surps.columns.get_level_values(1)
        avg_surps.to_csv(f"{args.data_dir}/avg_surps.csv", index=False)

    if args.metric_name == "delta_surps_by_tok":

        # Delta surprisal by token pos
        df["delta_surp"] = abs(df.surprisal - df.surprisal.shift(1))
        df.delta_surp.loc[(df.sentence_pos == 0) & (df.sentence_id == 0)] = np.nan
        delta_surps_by_tok = (
            df.dropna(subset=["delta_surp"])
            .groupby(["language", "variant", "sentence_len", "sentence_pos"])
            .agg({"delta_surp": [np.mean, sem, np.std]})
            .dropna()
            .reset_index()
        )
        delta_surps_by_tok.columns = delta_surps_by_tok.columns.get_level_values(
            0
        ) + delta_surps_by_tok.columns.get_level_values(1)
        (
            delta_surps_by_tok.query("sentence_len >= 10 & sentence_len <= 20").to_csv(
                f"{args.data_dir}/delta_surps_by_tok.csv", index=False
            )
        )

    if args.metric_name == "max_surps":

        # Max surprisal
        max_surps = (
            df.groupby(["language", "variant", "document_id", "sentence_id"])
            .surprisal.agg("max")
            .dropna()
            .reset_index()
            .groupby(["language", "variant"])
            .agg({"surprisal": [np.mean, sem, np.std]})
            .reset_index()
        )
        max_surps.columns = max_surps.columns.get_level_values(
            0
        ) + max_surps.columns.get_level_values(1)
        max_surps.to_csv(f"{args.data_dir}/max_surps.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    parser.add_argument("--data_dir")
    parser.add_argument("--metric_name")
    args = parser.parse_args()

    main(args)
