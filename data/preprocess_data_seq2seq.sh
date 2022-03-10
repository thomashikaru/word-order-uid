data_dir=$1
train_pref="train"
valid_pref="valid"
test_pref="test"

# for D in $(find $data_dir -mindepth 1 -maxdepth 1 -name "train*") ; do
langlist=("en")
langlist2=("enx")

for i in ${!langlist[@]}
do
   # prefix=$(basename $D |  cut -d '.' -f1) 
   fairseq-preprocess \
        --source-lang ${langlist[$i]} \
        --target-lang ${langlist2[$i]} \
        --trainpref $data_dir/$train_pref \
        --validpref $data_dir/$valid_pref \
        --testpref $data_dir/$test_pref \
        --destdir data-bin-seq2seq/${langlist[$i]} \
        --workers 20
   
   fairseq-preprocess \
        --source-lang ${langlist[$i]} \
        --target-lang ${langlist2[$i]} \
        --trainpref $data_dir-rev/$train_pref \
        --validpref $data_dir-rev/$valid_pref \
        --testpref $data_dir-rev/$test_pref \
        --destdir data-bin-seq2seq/${langlist[$i]}-rev \
	--workers 20
done
