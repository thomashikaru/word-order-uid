import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


def sent_to_hash(df):
    sent = " ".join(df["token"].astype(str))
    sum_surp = sum(df["surprisal"])
    variant = df["variant"].iloc[0]
    words = "".join(sorted(df["token"].astype(str)))
    s_hash = hash(words)
    return pd.DataFrame(
        {"sentence": [sent], "sum_surprisal": [sum_surp], "s_hash": [s_hash],}
    )


def pairgrid_heatmap(x, y, **kws):
    cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
    plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    ax = plt.gca()
    plt.hexbin(x, y, gridsize=50, cmap=cmap, extent=[0, 300, 0, 300], **kwargs)


def main():

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    sns.set(font_scale=2)

    df = pd.read_feather(
        "eval_results_cf_v5/cf_eval_data_8langs.feather",
        columns=[
            "surprisal",
            "token",
            "document_id",
            "language",
            "variant",
            "sentence_id",
        ],
        # dtype={"surprisal": np.float16, "token": str},
    )

    # df.to_csv("eval_results_cf_v4/cf_eval_data_8lang_hashes.csv", index=False)

    langs = ["en", "de", "fr", "ru", "vi", "hu", "tr", "id"]
    variants = [
        "Approx",
        "Real",
        "Efficient-OV",
        "Reverse",
        "RANDOM-1",
        "Sort-Freq",
        "Min-DL-Opt",
        "Min-DL-Loc",
    ]
    for lang in langs:

        df_sub = (
            df.query("language == @lang")
            .groupby(["language", "variant", "document_id", "sentence_id"])
            .apply(sent_to_hash)
            .reset_index()
            .groupby("s_hash")
            .apply(
                lambda x: x.assign(key=x.groupby("variant").cumcount()).pivot(
                    index="key", columns="variant", values="sum_surprisal"
                )
            )
            .reset_index()
        )

        plt.clf()
        # g = sns.pairplot(df_sub.drop_duplicates("s_hash"), vars=variants, kind="scatter", plot_kws={"alpha": 0.1})
        # g.set(xlim=(0, 300), ylim=(0, 300))
        # plt.savefig(f"eval_results_cf_v5/surp-pairplot-{lang}", dpi=180)
        # plt.savefig(f"eval_results_cf_v5/surp-pairplot-{lang}.svg", dpi=180)
        # plt.savefig(f"eval_results_cf_v5/surp-pairplot-{lang}.pdf", dpi=180)

        g = sns.PairGrid(df_sub.drop_duplicates("s_hash")[variants])
        g.map_offdiag(hexbin)
        g.map_diag(plt.hist, bins=50)
        plt.savefig(f"eval_results_cf_v5/surp-pairplot-heatmap-{lang}", dpi=180)
        plt.savefig(f"eval_results_cf_v5/surp-pairplot-heatmap-{lang}.svg", dpi=180)


if __name__ == "__main__":
    main()
