#!/bin/bash
rm paper_figures_04.log
source activate fenicsproject
nohup nice python -u -m paper_figures_04.py > paper_figures_04.log &
