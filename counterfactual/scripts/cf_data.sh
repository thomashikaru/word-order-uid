#!/bin/bash
LANGUAGE=$1
DATAFILE=$2
RESULTSFILE=$3

python ../apply_counterfactual_grammar.py --language $LANGUAGE --filename $DATAFILE > $RESULTSFILE