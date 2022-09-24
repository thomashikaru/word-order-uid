default=data-bin
ALL_DATA="${1:-$default}"
ALL_CHECKPOINTS="checkpoints_hypopt"
LOG_DIR="logs_hypopt"

module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy

langlist=("data-bin/ar" "data-bin/ar-rev" "data-bin/da" "data-bin/da-rev" "data-bin/el" "data-bin/el-rev" "data-bin/he" "data-bin/he-rev" "data-bin/hu" "data-bin/hu-rev" "data-bin/ko" "data-bin/ko-rev" "data-bin/sr" "data-bin/sr-rev" "data-bin/sv" "data-bin/sv-rev" "data-bin/th" "data-bin/th-rev")

langlist=("data-bin/ru" "data-bin/ru-rev")
for D in "${langlist[@]}" ; do
	prefix=$(basename $D |  cut -d '.' -f1)
	echo $D $ALL_CHECKPOINTS/$prefix
	bsub -W 24:00 -o $LOG_DIR/$prefix.out -R "select[gpu_mtotal0>=20480]"  -R "rusage[mem=30000,ngpus_excl_p=1]" bash train_model_transformer.sh $D $ALL_CHECKPOINTS/$prefix
done
