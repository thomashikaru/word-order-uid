#!/bin/bash

#langlist=("ar" "bg" "ca" "cs" "da" "de" "el" "en" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("ru" "vi" "en" "de" "fr")
extlist=("train" "test" "valid")
suff="x"

input_dir="wiki40b-txt-normalized"
# tmp_dir="wiki40b-txt-tokenized-seq2seq"
# tmp_dir_rev=$tmp_dir-rev
output_dir="wiki40b-txt-final-seq2seq"
output_dir_rev=$output_dir-rev

mkdir -p $tmp_dir $tmp_dir_rev
mkdir -p $output_dir $output_dir_rev

for lang in "${langlist[@]}"
do
    echo "Processing $lang"
    for ext in "${extlist[@]}"
    do
        python pytokenize.py --in_file $input_dir/$lang.$ext --out_file $output_dir/$ext.$lang --out_file2 $output_dir/$ext.$lang$suff --language $lang --seq2seq
        python pytokenize.py --in_file $input_dir/$lang.$ext --out_file $output_dir_rev/$ext.$lang --out_file2 $output_dir_rev/$ext.$lang$suff --language $lang --reverse --seq2seq
	echo "..."
    done
done

# printf -v joined_langlist '%s,' "${langlist[@]}"
# python sample.py \
#     --lang_code_list "${joined_langlist%,}" \
#     --input_prefix $tmp_dir \
#     --output_prefix $output_dir \
#     --ext_list "train.1,train.2,test.1,test.2,valid.1,valid.2"

# python sample.py \
#     --lang_code_list "${joined_langlist%,}" \
#     --input_prefix $tmp_dir_rev \
#     --output_prefix $output_dir_rev \
#     --ext_list "train.1,train.2,test.1,test.2,valid.1,valid.2"
