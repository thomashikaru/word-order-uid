#!/bin/bash
model_dir="../kenlm/build"
data_dir="wiki40b-txt-final"
out_file="ngram_scores.out"
langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("en")
#extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    echo $lang
    echo -e "language\t$lang" >> $out_file
	$model_dir/bin/lmplz -S 5G -o 5 < $data_dir/$lang.train > ngram/tmp.arpa && $model_dir/bin/query -v summary ngram/tmp.arpa < $data_dir/$lang.test >> $out_file
	echo -e "language\t$lang.rev" >> $out_file
        $model_dir/bin/lmplz -S 5G -o 5 < $data_dir-rev/$lang.train > ngram/tmp.arpa && $model_dir/bin/query -v summary ngram/tmp.arpa < $data_dir-rev/$lang.test >> $out_file
done
