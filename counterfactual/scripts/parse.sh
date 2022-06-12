#!/bin/bash
LANG=$1
DATA_DIR=$2
RESULTS_DIR=$3

python ../dep_parse.py --lang $LANG --data_dir $2 --parse_dir $3