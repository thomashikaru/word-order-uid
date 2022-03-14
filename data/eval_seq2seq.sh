DATA_DIR=$1
CHECKPOINT_DIR=$2
OUTPUT_DIR=$3

fairseq-generate $DATA_DIR \
    --path $CHECKPOINT_DIR \
    --batch-size 128 --beam 5 --remove-bpe \
    --results-path $OUTPUT_DIR