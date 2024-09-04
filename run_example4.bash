#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU

python3 example4_coupled_orthogonal_cracks.py -hmax 0.5 
python3 example4_coupled_orthogonal_cracks.py -hmax 0.25 
python3 example4_coupled_orthogonal_cracks.py -hmax 0.125

