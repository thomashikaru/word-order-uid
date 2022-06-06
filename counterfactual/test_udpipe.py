from ufal.udpipe import Model, Pipeline, ProcessingError
import sys

if __name__ == "__main__":

    # check arguments
    if len(sys.argv) < 4:
        sys.stderr.write(
            f"Usage: {sys.argv[0]} input_format(tokenize|conllu|horizontal|vertical) output_format(conllu) model_file\n"
        )
        sys.exit(1)

    sys.stderr.write("Loading model: ")
    model = Model.load(sys.argv[3])
    if not model:
        sys.stderr.write(f"Cannot load model from file '{sys.argv[3]}'\n")
        sys.exit(1)
    sys.stderr.write("done\n")

    pipeline = Pipeline(
        model, sys.argv[1], Pipeline.DEFAULT, Pipeline.DEFAULT, sys.argv[2]
    )
    error = ProcessingError()

    # Read whole input
    text = "".join(sys.stdin.readlines())

    # Process data
    processed = pipeline.process(text, error)
    if error.occurred():
        sys.stderr.write("An error occurred when running run_udpipe: ")
        sys.stderr.write(error.message)
        sys.stderr.write("\n")
        sys.exit(1)
    sys.stdout.write(processed)
