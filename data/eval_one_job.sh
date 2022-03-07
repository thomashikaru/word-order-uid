#!/bin/bash

default=data-bin
ALL_DATA="${1:-$default}"
default="checkpoints"
ALL_CHECKPOINTS="${2:-$default}"
default=1
NUM_SEEDS="${3:-$default}"


#langlist=("data-bin/ar" "data-bin/ar-rev" "data-bin/da" "data-bin/da-rev" "data-bin/el" "data-bin/el-rev" "data-bin/he" "data-bin/he-rev" "data-bin/hu" "data-bin/hu-rev" "data-bin/ko" "data-bin/ko-rev" "data-bin/sr" "data-bin/sr-rev" "data-bin/sv" "data-bin/sv-rev" "data-bin/th" "data-bin/th-rev")
langlist=("data-bin-bpe/en" "data-bin-bpe/en-rev")

for D in "${langlist[@]}" ; do
#for D in $(find $ALL_DATA -mindepth 1 -maxdepth 1 -type d) ; do
	prefix=$(basename $D |  cut -d '.' -f1)
	if test -f "$ALL_CHECKPOINTS/$prefix/checkpoint_best.pt"; then
		echo $prefix
		fairseq-eval-lm $D --path $ALL_CHECKPOINTS/$prefix/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 200
        fi
	if [[ $NUM_SEEDS -gt 1 ]]; then
	for i in $(seq 2 $NUM_SEEDS); do
    
		if test -f "$ALL_CHECKPOINTS/$prefix-$i/checkpoint_best.pt"; then
			echo $prefix-$i
    			fairseq-eval-lm $D --path $ALL_CHECKPOINTS/$prefix-$i/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 200	
		fi
	done
	fi
done

