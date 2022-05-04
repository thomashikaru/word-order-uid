DATA_DIR=$1
CHECKPOINT_DIR=$2
OUTPUT_DIR=$3

fairseq-generate $DATA_DIR \
    --path $CHECKPOINT_DIR \
    --max-tokens 4096 --beam 5 \
    --results-path $OUTPUT_DIR \
    --skip-invalid-size-inputs-valid-test \
    --score-reference
