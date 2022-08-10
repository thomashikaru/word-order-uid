import pandas as pd
import numpy as np
import glob

if __name__ == "__main__":

    filenames = glob.glob("freqs/*.csv")

    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        df["word"] = df["word"].astype(str)
        df = df[df["count"] > 1]
        df["word_lower"] = df["word"].apply(lambda x: x.lower())
        df = df.groupby("word_lower").sum().reset_index()
        dfs.append(df)

    df_all = pd.concat(dfs)
    df_all.rename(columns={"word_lower": "word"}, inplace=True)
    df_all.to_csv("freqs/freqs_8langs.csv", index=False)
