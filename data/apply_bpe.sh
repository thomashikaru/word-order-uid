BPE_CODES=bpe_codes/30k  # path where processed files will be stored
FASTBPE=../fastBPE/fast  # path to the fastBPE tool
INPUT_DIR="wiki40b-txt-final-doc"
INPUT_DIR_REV=$INPUT_DIR-rev
OUT_DIR="wiki40b-txt-final-doc-bpe"
OUT_DIR_REV=$OUT_DIR-rev

# create output path
mkdir -p $OUT_DIR $OUT_DIR_REV

langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("en" "de" "fr" "ru" "vi")
extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    echo $lang
    for ext in "${extlist[@]}"
    do
	#$FASTBPE applybpe $OUT_DIR/$lang.$ext $INPUT_DIR/$lang.$ext  $BPE_CODES/$lang.codes
	$FASTBPE applybpe $OUT_DIR_REV/$lang.$ext  $INPUT_DIR_REV/$lang.$ext $BPE_CODES/$lang.codes
done
done

