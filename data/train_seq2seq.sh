DATA_DIR=$1
SAVE_DIR=$2

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $DATA_DIR \
    --save-dir $SAVE_DIR \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion cross_entropy \
    --update-freq 100 \
    --max-tokens 4096 \
    --max-source-positions 512 \
    --max-target-positions 128 \
    --seed $RANDOM_SEED 5 \
    --skip-invalid-size-inputs-valid-test
