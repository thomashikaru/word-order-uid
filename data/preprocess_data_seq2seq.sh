data_dir=$1
train_pref="train"
valid_pref="valid"
test_pref="test"

# for D in $(find $data_dir -mindepth 1 -maxdepth 1 -name "train*") ; do
# langlist=("en" "fr" "de" "ru" "vi")
langlist=("en")
suff="x"

for lang in ${langlist[@]}
do
   # prefix=$(basename $D |  cut -d '.' -f1) 
   fairseq-preprocess \
        --source-lang $lang \
        --target-lang $lang$suff \
        --trainpref $data_dir/$train_pref \
        --validpref $data_dir/$valid_pref \
        --testpref $data_dir/$test_pref \
        --destdir data-bin-seq2seq/$lang \
        --workers 20
   
   fairseq-preprocess \
        --source-lang $lang \
        --target-lang $lang$suff \
        --trainpref $data_dir-rev/$train_pref \
        --validpref $data_dir-rev/$valid_pref \
        --testpref $data_dir-rev/$test_pref \
        --destdir data-bin-seq2seq/$lang-rev \
	--workers 20
done
