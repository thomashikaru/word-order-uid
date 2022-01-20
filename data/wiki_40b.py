import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
import argparse
import os

r1 = "_START_ARTICLE_\n[^_]*"
r2 = "_START_PARAGRAPH_\n"
r3 = "_START_SECTION_\n[^_]*"
r4 = "_NEWLINE_"

REGEX = re.compile(f"({r1}|{r2}|{r3}|{r4})")


def process_tf_dataset(ds, num_tokens, output_file):
    # Turn to a numpy df so that we can easily extract text
    numpy_items = tfds.as_numpy(ds)
    token_count = 0

    with open(output_file, "a") as f:
        for item in numpy_items:
            text = item.get("text").decode("UTF-8")
            text = re.sub(REGEX, " ", text)
            text = re.sub("\s+", " ", text).strip()
            f.write(text)
            f.write("\n")
            token_count += len(text.split())
            if token_count > num_tokens:
                break


def process_lang(lang_code, args):
    # Construct a tf.data.Dataset
    ds_train = tfds.load(f"wiki40b/{lang_code}", split="train", shuffle_files=True)
    process_tf_dataset(
        ds_train, args.num_train_tokens, args.output_prefix + lang_code + ".train"
    )

    ds_test = tfds.load(f"wiki40b/{lang_code}", split="test", shuffle_files=True)
    process_tf_dataset(
        ds_test, args.num_test_tokens, args.output_prefix + lang_code + ".test"
    )

    ds_valid = tfds.load(f"wiki40b/{lang_code}", split="validation", shuffle_files=True)
    process_tf_dataset(
        ds_valid, args.num_valid_tokens, args.output_prefix + lang_code + ".valid"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_code_list",
        help="comma-separated list of language codes to use, e.g. 'en,de,nl,hu'",
    )
    parser.add_argument(
        "--num_train_tokens",
        type=float,
        default=8e5,
        help="max number of training examples to sample",
    )
    parser.add_argument(
        "--num_test_tokens",
        type=float,
        default=1e5,
        help="max number of test examples to sample",
    )
    parser.add_argument(
        "--num_valid_tokens",
        type=float,
        default=1e5,
        help="max number of validation examples to sample",
    )
    parser.add_argument(
        "--output_prefix",
        default="wiki40b/",
        help="path to output destination for dataset",
    )
    args = parser.parse_args()

    dirname = os.path.dirname(args.output_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for lang_code in args.lang_code_list.split(","):
        process_lang(lang_code, args)

