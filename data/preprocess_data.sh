data_dir=$1
train_pref="train"
valid_pref="valid"
test_pref="test"
for D in $(find $data_dir -mindepth 1 -maxdepth 1 -type d) ; do
    fairseq-preprocess \
    	--only-source \
    	--trainpref $D/$train_pref \
    	--validpref $D/$valid_pref \
    	--testpref $D/$test_pref \
    	--destdir data-bin/$D \
    	--workers 20
    python reverse.py $D/$train_pref $D/$train_pref.rev
    python reverse.py $D/$valid_pref $D/$valid_pref.rev
    python reverse.py $D/$test_pref $D/$test_pref.rev
    fairseq-preprocess \
        --only-source \
        --trainpref $D/$train_pref.rev \
        --validpref $D/$valid_pref.rev \
        --testpref $D/$test_pref.rev \
        --destdir data-bin/$D-rev \
        --workers 20
done
