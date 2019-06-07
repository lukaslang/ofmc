#!/bin/bash
rm paper_figures_05.log
source activate fenicsproject
nohup nice python -u -m paper_figures_05.py > paper_figures_05.log &
