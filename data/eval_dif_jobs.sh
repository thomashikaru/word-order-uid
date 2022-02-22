default=data-bin
ALL_DATA="${1:-$default}"
ALL_CHECKPOINTS="checkpoints"
LOG_DIR="logs"

for D in $(find $ALL_DATA -mindepth 1 -maxdepth 1 -type d) ; do
	prefix=$(basename $D |  cut -d '.' -f1)
	reverse=(${prefix//-/ })
	if [ -z "${reverse[1]}" ]
	then
		bsub -o $LOG_DIR/per_example_perp2.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=60000,ngpus_excl_p=1]" python per_example_perp.py $ALL_CHECKPOINTS/$prefix $D wiki40b-txt-final/$prefix.test perps/$prefix.pt
		echo $ALL_CHECKPOINTS/$prefix $D wiki40b-txt-final/$prefix.test perps/$prefix.pt
	else
		bsub -o $LOG_DIR/per_example_perp2.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=60000,ngpus_excl_p=1]" python per_example_perp.py $ALL_CHECKPOINTS/$prefix $D wiki40b-txt-final-rev/${reverse[0]}.test perps/$prefix.pt
	echo $ALL_CHECKPOINTS/$prefix $D wiki40b-txt-final-rev/${reverse[0]}.test perps/$prefix.pt
	fi
done
