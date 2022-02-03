#!/bin/bash
python sample.py --lang_code_list "da,hu,ja"

langlist=("da" "hu" "ja")
extlist=("train" "test" "valid")

for lang in "${langlist[@]}"
do
    for ext in "${extlist[@]}"
    do
	mosesdecoder/scripts/tokenizer/normalize-punctuation.perl \
	    -l $lang \
	    < wiki40b-txt-sampled/$lang.$ext \
	    > wiki40b-txt-normalized/$lang.$ext
	mosesdecoder/scripts/tokenizer/tokenizer.perl \
	    -l $lang \
	    < wiki40b-txt-normalized/$lang.$ext \
	    > wiki40b-txt-tokenized/$lang.$ext
    done
done
