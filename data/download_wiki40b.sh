#!/bin/bash
langs="en"
# langs="fr,de,ru,es,en"
# langs="th,tr,bg,ca,cs,el,et,fa,fi,he,hi,hr,hu,id,lt,lv,ms,no,ro,sk,sl,sr,sv,tl,uk,vi,fr,de,ru,es,en"
python wiki_40b.py --lang_code_list $langs --data_dir "tfdata"
