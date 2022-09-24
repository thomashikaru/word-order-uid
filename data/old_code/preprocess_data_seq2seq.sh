# data_dir=$1
train_pref="train"
valid_pref="valid"
test_pref="test"

# DATA_DIR="wiki40b-txt-final-seq2seq-bpe"
DATA_DIR="wiki40b-txt-seq2seq-capital-sentencepiece"
DEST_DIR="data-bin-seq2seq-capital-sentencepiece"

# for D in $(find $data_dir -mindepth 1 -maxdepth 1 -name "train*") ; do
langlist=("en" "fr" "de" "ru" "vi")
# langlist=("en")
suff="x"

for lang in ${langlist[@]}
do
   # prefix=$(basename $D |  cut -d '.' -f1) 
   fairseq-preprocess \
        --source-lang $lang \
        --target-lang $lang$suff \
        --trainpref $DATA_DIR/$train_pref \
        --validpref $DATA_DIR/$valid_pref \
        --testpref $DATA_DIR/$test_pref \
        --destdir $DEST_DIR/$lang \
        --workers 20
   
   fairseq-preprocess \
        --source-lang $lang \
        --target-lang $lang$suff \
        --trainpref $DATA_DIR-rev/$train_pref \
        --validpref $DATA_DIR-rev/$valid_pref \
        --testpref $DATA_DIR-rev/$test_pref \
        --destdir $DEST_DIR/$lang-rev \
	--workers 20
done
