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

    # find the number of lines in the file
    N = 0
    with open(input_file) as f_in:
        for line in f_in:
            N += 1

    # get a sorted list of one million randomly sampled line indices
    idxs = sorted(random.sample(range(N), min(N, 1_000_000)))

    # iterate once through the list and print the lines whose indices were sampled
    # stop when the desired number of tokens is reached or the file ends
    tokens = 0
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        if len(idxs) == 0:
            return
        curr_index = idxs.pop(0)

        for i, line in enumerate(f_in):
            if curr_index == i:
                tokens += len(line.split())
                f_out.write(line)
                if tokens > num_tokens:
                    return
                if len(idxs) == 0:
                    return
                curr_index = idxs.pop(0)


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

