#!/bin/bash
# set -e
python l2r_train_eval.py $1 --savedir $2 --num-threads $3
python eval.py checkpoints/data.json eval_config.json
