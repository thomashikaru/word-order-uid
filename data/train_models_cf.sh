LOG_DIR="logs-cf"
ALL_CHECKPOINTS="checkpoints-cf-bpe"
datadir="data-bin-cf-bpe"

mkdir -p $LOG_DIR

for D in $(find $datadir -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D)
    echo $lang 
    
    for M in $(find $datadir/$D -mindepth 1 -maxdepth 1 -type d)
    do
        model=$(basename $M)
        echo $model
        mkdir -p $ALL_CHECKPOINTS/$lang/$model
        bsub -W 24:00 -o $LOG_DIR/$lang-$model.out \
            -R "select[gpu_mtotal0>=10000]"  \
            -R "rusage[mem=30000,ngpus_excl_p=1]" \
            bash train_model_transformer.sh $D/$model $ALL_CHECKPOINTS/$lang/$model
    done
done