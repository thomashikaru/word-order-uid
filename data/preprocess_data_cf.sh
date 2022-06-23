train_pref="train"
valid_pref="valid"
test_pref="test"

datadir="wiki40b-txt-cf-bpe"
destdir="data-bin-cf-bpe"

for D in $(find $datadir -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D) 
    for M in $(find $datadir/$D -mindepth 1 -maxdepth 1 -type d)
    do
        model=$(basename $M)
        fairseq-preprocess \
                --only-source \
                --trainpref $datadir/$lang/$model/$lang.$train_pref \
                --validpref $datadir/$lang/$model/$lang.$valid_pref \
                --testpref $datadir/$lang/$model/$lang.$test_pref \
                --destdir $destdir/$lang/$model \
                --bpe fastbpe \
                --workers 20
    done
done