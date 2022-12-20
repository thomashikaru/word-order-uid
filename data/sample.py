import itertools
import argparse
import random
import sys


def sample_data(input_file, output_file, num_tokens, seed):
    """Sample lines from input_file until the number of tokens
    exceeds num_tokens or the file is exhausted. 
    Save sampled lines to output_file.

    Args:
        input_file (str): input file path
        output_file (str): output file path
        num_tokens (int): threshold on number of tokens
    """
    random.seed(seed)
    with open(input_file) as f_in:
        lines = f_in.readlines()

    N = len(lines)
    idxs = random.sample(range(N), N)

    tokens = 0
    with open(output_file, "w") as f_out:
        for idx in idxs:
            line = lines[idx]
            tokens += len(line.split())
            f_out.write(line)
            if tokens > num_tokens:
                break


def cc100(args):
    lang_code_list = args.lang_code_list.split(",")
    ext_list = args.ext_list.split(",")

    for lang_code, ext in itertools.product(lang_code_list, ext_list):
        input_file = f"{args.input_prefix}/{lang_code}.txt"
        output_file = f"{args.output_prefix}/{lang_code}.{ext}"

        if ext == "train":
            sample_data(input_file, output_file, args.num_train_tokens, args.seed)
        else:
            sample_data(input_file, output_file, args.num_test_tokens, args.seed)


def wiki40b(args):
    lang_code_list = args.lang_code_list.split(",")
    ext_list = args.ext_list.split(",")

    for lang_code, ext in itertools.product(lang_code_list, ext_list):
        input_file = f"{args.input_prefix}/{lang_code}.{ext}"
        output_file = f"{args.output_prefix}/{lang_code}.{ext}"

        if ext == "train":
            sample_data(input_file, output_file, args.num_train_tokens, args.seed)
        else:
            sample_data(input_file, output_file, args.num_test_tokens, args.seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prefix")
    parser.add_argument("--output_prefix")
    parser.add_argument("--num_train_tokens", type=int, default=20_000_000)
    parser.add_argument("--num_test_tokens", type=int, default=1_000_000)
    parser.add_argument("--lang_code_list")
    parser.add_argument("--ext_list", default="train,test,valid")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cc100", action="store_true")
    args = parser.parse_args()

    if args.cc100:
        cc100(args)

    if args.wiki40b:
        wiki40b(args)

