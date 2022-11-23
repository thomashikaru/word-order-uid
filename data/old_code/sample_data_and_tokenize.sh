#!/bin/bash
python sample.py --lang_code_list "ar,bg,ca,cs,de,el,es,et,fa,fi,fr,he,hi,hr,id,it,ko,lt,lv,ms,nl,no,pl,pt,ro,ru,sk,sl,sr,sv,th,tl,tr,uk,vi,zh-cn"

# langlist=("da" "hu" "ja")
langlist=("ar" "bg" "ca" "cs" "de" "el" "es" "et" "fa" "fi" "fr" "he" "hi" "hr" "id" "it" "ko" "lt" "lv" "ms" "nl" "no" "pl" "pt" "ro" "ru" "sk" "sl" "sr" "sv" "th" "tl" "tr" "uk" "vi" "zh-cn")
extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    for ext in "${extlist[@]}"
    do
	mosesdecoder/scripts/tokenizer/normalize-punctuation.perl \
	    -l $lang \
	    < wiki40b-txt-sampled/$lang.$ext \
	    > wiki40b-txt-normalized/$lang.$ext
	#mosesdecoder/scripts/tokenizer/tokenizer.perl \
	#    -l $lang \
	#    < wiki40b-txt-normalized/$lang.$ext \
	#    > wiki40b-txt-tokenized/$lang.$ext
    done
done
