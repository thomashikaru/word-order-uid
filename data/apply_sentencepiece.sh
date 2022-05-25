
SPM_ENCODE="../scripts/spm_encode.py"
SP_DIR="sentencepiece/16k"
DATA_DIR="wiki40b-txt-seq2seq-capital"
OUT_DIR="wiki40b-txt-seq2seq-capital-sentencepiece"

langlist=("en" "de" "fr" "ru" "vi")
extlist=("train" "test" "valid")
suff="x"

TRAIN_MINLEN=1
TRAIN_MAXLEN=1024

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR-rev

# encode train/valid
echo "encoding train with learned BPE..."
for lang in "${langlist[@]}"
do
    for ext in "${extlist[@]}"
    do
        python "$SPM_ENCODE" \
            --model "$SP_DIR/sentencepiece-$lang.bpe.model" \
            --output_format=piece \
            --inputs $DATA_DIR/$ext.$lang $DATA_DIR-rev/$ext.$lang $DATA_DIR/$ext.$lang$suff $DATA_DIR-rev/$ext.$lang$suff \
            --outputs $OUT_DIR/$ext.$lang $OUT_DIR-rev/$ext.$lang $OUT_DIR/$ext.$lang$suff $OUT_DIR-rev/$ext.$lang$suff \
            --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
    done
done

