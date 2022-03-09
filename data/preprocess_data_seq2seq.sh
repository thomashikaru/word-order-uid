data_dir=$1
train_pref="train"
valid_pref="valid"
test_pref="test"

for D in $(find $data_dir -mindepth 1 -maxdepth 1 -name "*.train*") ; do
   prefix=$(basename $D |  cut -d '.' -f1) 
   fairseq-preprocess \
        --only-source \
        --trainpref $data_dir/$prefix.$train_pref \
        --validpref $data_dir/$prefix.$valid_pref \
        --testpref $data_dir/$prefix.$test_pref \
        --destdir data-bin-seq2seq/$prefix \
        --bpe fastbpe \
        --workers 20
   
   fairseq-preprocess \
        --only-source \
        --trainpref $data_dir-rev/$prefix.$train_pref \
        --validpref $data_dir-rev/$prefix.$valid_pref \
        --testpref $data_dir-rev/$prefix.$test_pref \
        --destdir data-bin-seq2seq/$prefix-rev \
        --bpe fastbpe \
	    --workers 20
done