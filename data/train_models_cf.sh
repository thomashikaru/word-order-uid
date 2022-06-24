# This script trains language models for a variety of languages and real/counterfactual grammars.
# There are currently 5 languages (en, ru, fr, de, vi) and 4 grammars (real, random, optimized VO, optimized OV)
# Training logs for English-RANDOM would be in logs-cf/en-RANDOM.out
# Checkpoints for English-RANDOM would be in checkpoints-cf-bpe/en/RANDOM/

LOG_DIR="logs-cf"
ALL_CHECKPOINTS="checkpoints-cf-bpe"
datadir="data-bin-cf-bpe"

module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

mkdir -p $LOG_DIR

for D in $(find $datadir -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D)
    echo $lang 
    
    for M in $(find $datadir/$lang -mindepth 1 -maxdepth 1 -type d)
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
