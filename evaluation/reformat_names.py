import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import glob
from variant_name2id import name2id

mapping = {
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
    "EFFICIENT_OV": "Efficient-OV",
    "EFFICIENT_VO": "Efficient-VO",
    "APPROX": "Approx",
    "MIN_DL_OPT": "Min-DL-Opt",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    args = parser.parse_args()
    df = pd.read_csv(args.inputfile)
    df.variant = df.variant.replace(mapping)
    df.to_csv(args.inputfile)
