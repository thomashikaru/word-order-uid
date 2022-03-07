default=data-bin
ALL_DATA="${1:-$default}"
default="logs"
LOG_DIR="${2:-$default}"
default="checkpoints"
ALL_CHECKPOINTS="${3:-$default}"
default=1
NUM_SEEDS="${4:-$default}"


langlist=("data-bin-bpe/cs" "data-bin-bpe/cs-rev" "data-bin-bpe/de" "data-bin-bpe/de-rev" "data-bin-bpe/et" "data-bin-bpe/et-rev" "data-bin-bpe/fi" "data-bin-bpe/fi-rev" "data-bin-bpe/fr" "data-bin-bpe/fr-rev" "data-bin-bpe/he" "data-bin-bpe/he-rev" "data-bin-bpe/pl" "data-bin-bpe/pl-rev" "data-bin-bpe/ru" "data-bin-bpe/ru-rev" "data-bin-bpe/sr" "data-bin-bpe/sr-rev" "data-bin-bpe/sk" "data-bin-bpe/sk-rev" "data-bin-bpe/sl" "data-bin-bpe/sl-rev" "data-bin-bpe/th" "data-bin-bpe/th-rev" "data-bin-bpe/uk" "data-bin-bpe/uk-rev")
langlist=("data-bin-bpe/en" "data-bin-bpe/en-rev")
for D in "${langlist[@]}" ; do
#for D in $(find $ALL_DATA -mindepth 1 -maxdepth 1 -type d) ; do
	prefix=$(basename $D |  cut -d '.' -f1)
	echo $D $ALL_CHECKPOINTS/$prefix
	for i in $(seq 2 $NUM_SEEDS); do  
		bsub -W 24:00 -o $LOG_DIR/$prefix-$i.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=30000,ngpus_excl_p=1]" bash train_model_transformer.sh $D $ALL_CHECKPOINTS/$prefix-$i $i
done
done
