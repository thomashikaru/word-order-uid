# This script evaluates trained language models for a variety of languages and real/counterfactual grammars.
# It outputs token lists and per-word logprobs
# There are currently 8 languages (en, ru, fr, de, vi, tr, id, hu) and 10 grammars/variants.
# Evaluation logs for English-RANDOM would be in logs-cf-eval/en-RANDOM.out
# Tokens and Logprobs for English-RANDOM would be in perps-cf/en-RANDOM.pt

DATA_BIN="data-bin-cf-bpe-v6"
CHECKPOINTS="checkpoints-cf-bpe-v6"
LOG_DIR="logs-cf-eval-v6"
OUT_DIR="perps-cf-v6"
DATA_DIR="wiki40b-txt-cf-bpe-v6"

module load gcc/6.3.0
module load python_gpu/3.8.5 hdf5 eth_proxy
module load geos
module load libspatialindex

mkdir -p $LOG_DIR
mkdir -p $OUT_DIR

for D in $(find $DATA_BIN -mindepth 1 -maxdepth 1 -type d)
do
    lang=$(basename $D)
    echo $lang 
    
    for M in $(find $DATA_BIN/$lang -mindepth 1 -maxdepth 1 -type d)
    do
        model=$(basename $M)
        echo $model
        bsub -W 4:00 -o $LOG_DIR/$lang-$model.out \
            -R "select[gpu_mtotal0>=10000]"  \
            -R "rusage[mem=60000,ngpus_excl_p=1]" \
            python per_example_perp.py $CHECKPOINTS/$lang/$model $D/$model $DATA_DIR/$lang/$model/$lang.test $OUT_DIR/$lang-$model.pt
    done
done