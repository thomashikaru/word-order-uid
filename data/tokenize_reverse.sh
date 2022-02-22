#!/bin/bash

#langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
langlist=("da" "hu" "ja")
extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    for ext in "${extlist[@]}"
    do
	python pytokenize.py wiki40b-txt-normalized/$lang.$ext wiki40b-txt-tokenized/$lang.$ext $lang
	
	python reverse.py wiki40b-txt-normalized/$lang.$ext wiki40b-txt-tokenized-rev/$lang.$ext $lang

done
done 

python sample.py \
    --lang_code_list "ar,bg,ca,cs,da,de,el,es,en,et,fa,fi,fr,he,hi,hr,hu,id,it,ja,ko,lt,lv,ms,nl,no,pl,pt,ro,ru,sk,sl,sr,sv,th,tl,tr,uk,vi,zh-cn" \
    --input_prefix "wiki40b-txt-tokenized" \
    --output_prefix "wiki40b-txt-final"

python sample.py \
    --lang_code_list "ar,bg,ca,cs,da,de,el,es,en,et,fa,fi,fr,he,hi,hr,hu,id,it,ja,ko,lt,lv,ms,nl,no,pl,pt,ro,ru,sk,sl,sr,sv,th,tl,tr,uk,vi,zh-cn" \
    --input_prefix "wiki40b-txt-tokenized-rev" \
    --output_prefix "wiki40b-txt-final-rev"
