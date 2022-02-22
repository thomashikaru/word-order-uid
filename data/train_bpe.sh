OUTPATH=bpe_codes/30k  # path where processed files will be stored
FASTBPE=../../fastBPE/fast  # path to the fastBPE tool
NUM_OPS=30000

# create output path
mkdir -p $OUTPATH

langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    echo $lang
	# learn bpe codes on the training set (or only use a subset of it)
	$FASTBPE learnbpe $NUM_OPS wiki40b-txt-final/$lang.train > $OUTPATH/$lang.codes

    for ext in "${extlist[@]}"
    do
	$FASTBPE applybpe wiki40b-txt-final-bpe/$lang.$ext wiki40b-txt-final/$lang.$ext  $OUTPATH/$lang.codes
	$FASTBPE applybpe wiki40b-txt-final-bpe-rev/$lang.$ext  wiki40b-txt-final-rev/$lang.$ext $OUTPATH/$lang.codes
done
done

