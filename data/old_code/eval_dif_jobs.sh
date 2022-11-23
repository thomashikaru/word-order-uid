default=data-bin-doc
DATA_BIN="${1:-$default}"
CHECKPOINTS="checkpoints-doc"
LOG_DIR="logs/doc"
OUT_DIR="perps/doc"
DATA_DIR="wiki40b-txt-tokenized-doc-bpe"

for D in $(find $DATA_BIN -mindepth 1 -maxdepth 1 -type d) ; do
	prefix=$(basename $D |  cut -d '.' -f1)
	reverse=(${prefix//-/ })
	if [ -f $OUT_DIR/$prefix.pt ]; then
    		continue
  	fi
	if [ -z "${reverse[1]}" ]
	then
		bsub -W 4:00 -o $LOG_DIR/per_example_perp.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=60000,ngpus_excl_p=1]" python per_example_perp.py $CHECKPOINTS/$prefix $D $DATA_DIR/$prefix.test $OUT_DIR/$prefix.pt
		echo $CHECKPOINTS/$prefix $D $DATA_DIR/$prefix.test $OUT_DIR/$prefix.pt
	else
		bsub -W 4:00 -o $LOG_DIR/per_example_perp.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=60000,ngpus_excl_p=1]" python per_example_perp.py $CHECKPOINTS/$prefix $D $DATA_DIR-rev/${reverse[0]}.test $OUT_DIR/$prefix.pt
	echo $CHECKPOINTS/$prefix $D $DATA_DIR-rev/${reverse[0]}.test $OUT_DIR/$prefix.pt
	fi
done
