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
    "RANDOM_1": "Random_1",
    "RANDOM_2": "Random_2",
    "RANDOM_3": "Random_3",
    "RANDOM_4": "Random_4",
    "RANDOM_5": "Random_5",
    "SORT_FREQ": "Sort_Freq",
    "SORT_FREQ_REV": "Sort_Freq_Rev",
    "REVERSE": "Reverse",
    "MIN_DL_PROJ": "Min_DL_Loc",
    "EFFICIENT_OV": "Efficient_OV",
    "EFFICIENT_VO": "Efficient_VO",
    "APPROX": "Approx",
    "MIN_DL_OPT": "Min_DL_Opt",
    "Random-1": "Random_1",
    "Random-2": "Random_2",
    "Random-3": "Random_3",
    "Random-4": "Random_4",
    "Random-5": "Random_5",
    "Sort-Freq": "Sort_Freq",
    "Sort-Freq-Rev": "Sort_Freq_Rev",
    "Reverse": "Reverse",
    "Min-DL-Loc": "Min_DL_Loc",
    "Efficient-OV": "Efficient_OV",
    "Efficient-VO": "Efficient_VO",
    "Min-DL-Opt": "Min_DL_Opt",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile")
    args = parser.parse_args()
    df = pd.read_csv(args.inputfile)
    df.variant = df.variant.replace(mapping)
    df.to_csv(args.inputfile)
