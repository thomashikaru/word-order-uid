OUTPATH=bpe_codes_cf/30k  # path where processed files will be stored
FASTBPE=../fastBPE/fast  # path to the fastBPE tool
NUM_OPS=30000
INPUT_DIR="wiki40b-txt-cf"

# create output path
mkdir -p $OUTPATH

# langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("en" "de" "fr" "ru" "vi")

for lang in "${langlist[@]}"
do
    echo $lang
    cat $INPUT_DIR/$lang/*/$lang.train > $lang-agg.txt
    # learn bpe codes on the training set (or only use a subset of it)
    $FASTBPE learnbpe $NUM_OPS $lang-agg.txt > $OUTPATH/$lang.codes
    rm $lang-agg.txt
done
