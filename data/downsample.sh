#!/bin/bash
python sample.py \
    --lang_code_list "ar,bg,ca,cs,da,de,el,es,en,et,fa,fi,fr,he,hi,hr,hu,id,it,ja,ko,lt,lv,ms,nl,no,pl,pt,ro,ru,sk,sl,sr,sv,th,tl,tr,uk,vi,zh-cn" \
    --input_prefix "wiki40b-txt-tokenized" \
    --output_prefix "wiki40b-txt-final"
