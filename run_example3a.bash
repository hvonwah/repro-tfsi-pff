#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results

python3 example3a_propagating_pff_reconstr.py -hmax 0.5    -eps 0.01
python3 example3a_propagating_pff_reconstr.py -hmax 0.25   -eps 0.01
python3 example3a_propagating_pff_reconstr.py -hmax 0.125  -eps 0.01
