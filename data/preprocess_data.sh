data_dir=$1
train_pref="train"
valid_pref="valid"
test_pref="test"

langlist=("wiki40b-txt-final-bpe/en.train")
for D in "${langlist[@]}" ; do
#for D in $(find $data_dir -mindepth 1 -maxdepth 1 -name "*.train") ; do
   prefix=$(basename $D |  cut -d '.' -f1) 
   fairseq-preprocess \
    	--only-source \
    	--trainpref $data_dir/$prefix.$train_pref \
    	--validpref $data_dir/$prefix.$valid_pref \
    	--testpref $data_dir/$prefix.$test_pref \
    	--destdir data-bin-bpe/$prefix \
    	--bpe fastbpe \
	--workers 20
    
   fairseq-preprocess \
        --only-source \
        --trainpref $data_dir-rev/$prefix.$train_pref \
        --validpref $data_dir-rev/$prefix.$valid_pref \
        --testpref $data_dir-rev/$prefix.$test_pref \
        --destdir data-bin-bpe/$prefix-rev \
        --bpe fastbpe \
	--workers 20
done
