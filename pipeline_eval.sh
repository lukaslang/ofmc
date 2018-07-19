#!/bin/bash
rm pipeline_eval.log
source activate fenicsproject
nohup nice python -u -m pipeline_eval.py > pipeline_eval.log &
