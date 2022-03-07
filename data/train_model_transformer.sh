DATA_DIR=$1
SAVE_DIR=$2
default=1
RANDOM_SEED="${3:-$default}"

fairseq-train --task language_modeling \
	  $DATA_DIR \
	  --save-dir $SAVE_DIR \
	  --arch transformer_lm --share-decoder-input-output-embed \
	  --dropout 0.1 \
	  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
	  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
	  --tokens-per-sample 512 --sample-break-mode none \
	  --max-tokens 512 --update-freq 64 \
	  --fp16 \
	  --max-update 50000 --max-epoch 35 --patience 3  \
  	  --seed $RANDOM_SEED	  

