#!/bin/bash
LANGUAGE=$1
MODEL=$2
DATAFILE=$3
RESULTSFILE=$4
SEED=$5

python ../apply_counterfactual_grammar.py --output_dl_only --language $LANGUAGE --model $MODEL --filename $DATAFILE --seed $SEED > $RESULTSFILE