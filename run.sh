#!/bin/bash -x
function train () {
python3 train.py \
--gpu 0 \
--data ./final_data/
}
train
