#!/bin/bash
LANGUAGE=$1
MODEL=$2
DATAFILE=$3

python ../apply_counterfactual_grammar.py --language $LANGUAGE --model $MODEL --filename $DATAFILE --freq_opt
