#!/bin/bash

#langlist=("ar" "bg" "ca" "cs" "da" "de" "el" "en" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("ru" "vi") #"en" "de" "fr" 
extlist=("train" "test" "valid")

input_dir="/data/word-order-data/wiki40b-txt-normalized"
tmp_dir="wiki40b-txt-tokenized-sent"
tmp_dir_rev=$tmp_dir-rev
output_dir="wiki40b-txt-final-sent"
output_dir_rev=$output_dir-rev

mkdir -p $tmp_dir $tmp_dir_rev
mkdir -p $output_dir $output_dir_rev

for lang in "${langlist[@]}"
do
    echo "Processing $lang"
    for ext in "${extlist[@]}"
    do
        python pytokenize.py --in_file $input_dir/$lang.$ext --out_file $tmp_dir/$lang.$ext --language $lang --split_sentences

        python pytokenize.py --in_file $input_dir/$lang.$ext --out_file $tmp_dir_rev/$lang.$ext --language $lang --reverse --split_sentences

done
done

printf -v joined_langlist '%s,' "${langlist[@]}"
python sample.py \
    --lang_code_list "${joined_langlist%,}" \
    --input_prefix $tmp_dir \
    --output_prefix $output_dir

python sample.py \
    --lang_code_list "${joined_langlist%,}" \
    --input_prefix $tmp_dir_rev \
    --output_prefix $output_dir_rev
