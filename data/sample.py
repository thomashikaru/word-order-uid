import itertools
import argparse
import random


def sample_data(input_file, output_file, num_tokens):
    """Sample lines from input_file until the number of tokens
    exceeds num_tokens or the file is exhausted. 
    Save sampled lines to output_file.

    Args:
        input_file (str): input file path
        output_file (str): output file path
        num_tokens (int): threshold on number of tokens
    """
    with open(input_file) as f_in:
        lines = f_in.readlines()

    N = len(lines)
    idxs = random.sample(range(N), N)

    tokens = 0
    with open(output_file, "w") as f_out:
        for idx in idxs:
            line = lines[idx]
            tokens += len(line.split())
            f_out.write(line + "\n")
            if tokens > num_tokens:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prefix", default="wiki40b-txt")
    parser.add_argument("--output_prefix", default="wiki40b-txt-sampled")
    parser.add_argument("--num_train_tokens", type=int, default=20_000_000)
    parser.add_argument("--num_test_tokens", type=int, default=1_000_000)
    parser.add_argument("--lang_code_list")
    parser.add_argument("--ext_list", default="train,test,valid")
    args = parser.parse_args()

    lang_code_list = args.lang_code_list.split(",")
    ext_list = args.ext_list.split(",")

    for lang_code, ext in itertools.product(lang_code_list, ext_list):
        input_file = f"{args.input_prefix}/{lang_code}.{ext}"
        output_file = f"{args.output_prefix}/{lang_code}.{ext}"

        if ext == "train":
            sample_data(input_file, output_file, args.num_train_tokens)
        else:
            sample_data(input_file, output_file, args.num_test_tokens)

