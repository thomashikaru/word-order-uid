#!/bin/bash
LANGUAGE=$1
MODEL=$2
DATAFILE=$3
RESULTSFILE=$4

python ../apply_counterfactual_grammar.py --language $LANGUAGE --model $MODEL --filename $DATAFILE > $RESULTSFILE
