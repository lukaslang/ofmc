#!/bin/bash
rm pipeline_eval_figures.log
source activate fenicsproject
nohup nice python -u -m pipeline_eval_figures.py > pipeline_eval_figures.log &
