#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results

python3 example2_stationary_coupled_pff_tfsi.py -hmax 0.16 -alpha_th 1e-5 -theta 0

