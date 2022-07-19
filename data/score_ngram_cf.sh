#!/bin/bash
lang=$1
model=$2
data_dir=$3
out_file=$4

model_dir="../kenlm/build"

$model_dir/bin/lmplz -S 5G -o 5 < $data_dir/$lang/$model/$lang.train > $lang.$model.arpa && \
            $model_dir/bin/query -v summary $lang.$model.arpa < $data_dir/$lang/$model/$lang.test > $out_file
rm $lang.$model.arpa
