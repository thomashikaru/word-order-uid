#!/bin/bash
model_dir="../kenlm/build"
data_dir="wiki40b-txt-cf-v2"
out_file="ngram_scores_cf.out"
langlist=("en" "ru" "fr" "de" "vi")


for D in $(find $data_dir -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D)
    echo $lang 
    
    for M in $(find $data_dir/$lang -mindepth 1 -maxdepth 1 -type d)
    do
        model=$(basename $M)
        echo -e "\t$model"
        echo -e "language\t$lang\t$model" >> $out_file
        $model_dir/bin/lmplz -S 5G -o 5 < $data_dir/$lang/$model/$lang.train > tmp_cf.arpa && \
            $model_dir/bin/query -v summary tmp_cf.arpa < $data_dir/$lang/$model/$lang.test >> $out_file
    done
done
