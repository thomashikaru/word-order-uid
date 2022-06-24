DATA_BIN="data-bin-cf-bpe"
CHECKPOINTS="checkpoints-cf-bpe"
LOG_DIR="logs-cf-eval"
OUT_DIR="perps-cf"
DATA_DIR="wiki40b-txt-cf-bpe"

module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

mkdir -p $LOG_DIR

# for D in $(find $DATA_BIN -mindepth 1 -maxdepth 1 -type d) ; do
# 	prefix=$(basename $D |  cut -d '.' -f1)
# 	reverse=(${prefix//-/ })
# 	if [ -f $OUT_DIR/$prefix.pt ]; then
#     		continue
#   	fi
# 	if [ -z "${reverse[1]}" ]
# 	then
# 		bsub -W 4:00 -o $LOG_DIR/per_example_perp.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=60000,ngpus_excl_p=1]" python per_example_perp.py $CHECKPOINTS/$prefix $D $DATA_DIR/$prefix.test $OUT_DIR/$prefix.pt
# 		echo $CHECKPOINTS/$prefix $D $DATA_DIR/$prefix.test $OUT_DIR/$prefix.pt
# 	else
# 		bsub -W 4:00 -o $LOG_DIR/per_example_perp.out -R "select[gpu_mtotal0>=10000]"  -R "rusage[mem=60000,ngpus_excl_p=1]" python per_example_perp.py $CHECKPOINTS/$prefix $D $DATA_DIR-rev/${reverse[0]}.test $OUT_DIR/$prefix.pt
# 	echo $CHECKPOINTS/$prefix $D $DATA_DIR-rev/${reverse[0]}.test $OUT_DIR/$prefix.pt
# 	fi
# done

for D in $(find $DATA_BIN -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D)
    echo $lang 
    
    for M in $(find $DATA_BIN/$lang -mindepth 1 -maxdepth 1 -type d)
    do
        model=$(basename $M)
        echo $model
        mkdir -p $OUT_DIR/$lang/$model
        bsub -W 4:00 -o $LOG_DIR/$lang-$model.out \
            -R "select[gpu_mtotal0>=10000]"  \
            -R "rusage[mem=60000,ngpus_excl_p=1]" \
            python per_example_perp.py $CHECKPOINTS/$lang/$model $D/$model $DATA_DIR/$lang/$model/$lang.test $OUT_DIR/$lang-$model.pt
    done
done