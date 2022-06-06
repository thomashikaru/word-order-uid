from ufal.udpipe import Model, Pipeline, ProcessingError
import sys
import argparse
import numpy as np
import pandas as pd
import os
from mosestokenizer import MosesTokenizer, MosesSentenceSplitter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang", help="2-letter language code such as en, ru, vi, etc.", default="en"
    )
    parser.add_argument(
        "--udpipe_model_path",
        help="path to UDPipe model file for this language",
        default="udpipe_models/english-lines-ud-2.5-191206.udpipe",
    )
    parser.add_argument(
        "--data_dir",
        help="path to data directory with original (normal-order) text",
        default=".",
    )
    parser.add_argument(
        "--parse_dir",
        help="path to directory where CONLLU parses of sentences should be stored",
        default=".",
    )
    parser.add_argument(
        "--partitions",
        default="train,test,valid",
        help="comma-seprated list of partitions",
    )
    args = parser.parse_args()

    if not os.path.exists(args.parse_dir):
        os.system(f"mkdir -p {args.parse_dir}")

    # load UDPipe Model
    sys.stderr.write("Loading model: ")
    model = Model.load(args.udpipe_model_path)
    if not model:
        sys.stderr.write(f"Cannot load model from file '{args.udpipe_model_path}'\n")
        sys.exit(1)
    sys.stderr.write("done\n")

    # create pipeline
    pipeline = Pipeline(
        model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )
    error = ProcessingError()

    # Make sentence tokenizer
    sent_tokenize = MosesSentenceSplitter(args.lang)
    word_tokenize = MosesTokenizer(args.lang)

    for partition in args.partitions.split(","):
        input_path = os.path.join(args.data_dir, f"{args.lang}.{partition}")
        output_path = os.path.join(args.parse_dir, f"{args.lang}.{partition}.conllu")

        with open(input_path) as f:
            documents = f.readlines()

        with open(output_path, "w") as f:

            for i, document in enumerate(documents):

                # split sentences
                sentences = sent_tokenize([document])

                sentences_tokenized = [word_tokenize(s) for s in sentences]
                sentences = [" ".join(s) for s in sentences_tokenized]
                sentences = "\n".join(sentences)

                # Process data
                processed = pipeline.process(sentences, error)
                if error.occurred():
                    sys.stderr.write("An error occurred in run_udpipe: ")
                    sys.stderr.write(error.message)
                    sys.stderr.write("\n")
                    sys.exit(1)

                f.write(processed)
