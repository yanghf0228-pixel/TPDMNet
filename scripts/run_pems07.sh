#!/usr/bin/env bash
set -e
python run.py --dataset PEMSD7 --type P --model TPDMNet --mode train --seed 10 --device cuda:0
