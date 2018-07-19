#!/bin/bash
source activate fenicsproject
nohup nice python -m pipeline_eval.py > pipeline_eval.log 2>&1
