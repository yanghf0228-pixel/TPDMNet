#!/usr/bin/env bash
set -e
python run.py --dataset PEMSD8 --type P --model TPDMNet --mode train --seed 10 --device cuda:0
