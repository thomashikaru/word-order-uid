DATA_DIR=$1
SAVE_DIR=$2

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $DATA_DIR \
    --save-dir $SAVE_DIR \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --truncate-source \
    --max-source-positions 1024 \
    --max-target-positions 128 \
    --maximize-best-checkpoint-metric \
    --seed $RANDOM_SEED 5
