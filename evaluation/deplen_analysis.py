import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import glob
from scipy.stats import zscore


def plot_dl_vs_surp():
    """This is part of a secondary analysis that was not included in the paper.
    Briefly, it looks at the relationship between average dependency length and
    surprisal, but only does this for some languages currently.
    """
    filenames = glob.glob("../data/wiki40b-txt-cf-deplens/*/*/testset_deplens.txt")
    data = []

    for filename in filenames:
        _, _, lang, variant, _ = filename.split("/")
        with open(filename) as f:
            text = f.read()
            sentlen = float(text.split()[-5])
            deplen = float(text.split()[-1])
        data.append(
            {"lang": lang, "variant": variant, "deplen": deplen, "sentlen": sentlen}
        )

    df = pd.DataFrame(data)

    variants = [
        "REAL_REAL",
        "REVERSE",
        "APPROX",
        "EFFICIENT_OV",
        "EFFICIENT_VO",
        "MIN_DL_OPT",
        "MIN_DL_PROJ",
        "RANDOM_1",
        "RANDOM_2",
        "RANDOM_3",
        "RANDOM_4",
        "RANDOM_5",
        "SORT_FREQ",
        "SORT_FREQ_REV",
    ]
    colors = [
        "gray",
        "red",
        "orange",
        "blue",
        "blue",
        "green",
        "green",
        "gold",
        "gold",
        "gold",
        "gold",
        "gold",
        "purple",
        "purple",
    ]

    color_mapping = dict(zip(variants, colors))

    g = sns.FacetGrid(data=df, col="lang", col_wrap=5, height=4)
    g.map_dataframe(
        sns.barplot,
        x="variant",
        y="deplen",
        hue="variant",
        dodge=False,
        hue_order=variants,
        order=variants,
        palette=color_mapping,
    )
    g.set_xticklabels(rotation=45, ha="right")
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
    sns.regplot(data=surps, x="dl_z", y="surp_z", hue="lang")
    plt.savefig("surpz-vs-dlz-reg", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    plot_dl_vs_surp()
