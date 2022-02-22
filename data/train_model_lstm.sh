DATA_DIR=$1
SAVE_DIR=$2
default=1
RANDOM_SEED="${3:-$default}"

fairseq-train --task language_modeling \
  $DATA_DIR \
  --save-dir $SAVE_DIR \
  --save-interval-updates 10000 \
  --keep-interval-updates 10 \
  --keep-last-epochs 10 \
  --arch lstm_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --decoder-layers 2 --decoder-hidden-size 1024 --decoder-embed-dim 400 --decoder-out-embed-dim 400 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0000012 --clip-norm 0.0  \
  --lr  0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 512 --update-freq 10 --seed $RANDOM_SEED \
  --log-format json --log-interval 10 --max-update 100000
