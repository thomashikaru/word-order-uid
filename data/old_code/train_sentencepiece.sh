OUTPATH=sentencepiece/16k
INPUT_DIR="wiki40b-txt-seq2seq-capital"
BPESIZE=16384

SCRIPTS=../scripts
SPM_TRAIN=$SCRIPTS/spm_train.py

# create output path
mkdir -p $OUTPATH

# langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("en" "de" "fr" "ru" "vi")

for lang in "${langlist[@]}"
do
    echo $lang

    # TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.${SRC}-${TGT}.${SRC}; echo $DATA/train.${SRC}-${TGT}.${TGT}; done | tr "\n" ",")
    TRAIN_FILES="$INPUT_DIR/train.${lang}x,$INPUT_DIR-rev/train.${lang}x"
    
    echo "learning joint BPE over ${TRAIN_FILES}..."

    python "$SPM_TRAIN" \
        --input=$TRAIN_FILES \
        --model_prefix=$OUTPATH/sentencepiece-$lang.bpe \
        --vocab_size=$BPESIZE \
        --character_coverage=1.0 \
        --model_type=bpe

done