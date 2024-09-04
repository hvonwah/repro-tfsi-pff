#!/bin/bash

source /mnt/local/Applications/ngs-venv/bin/activate
export MKL_THREADING_LAYER=GNU
mkdir -p results

python3 example1_thermal_phase_field_fracture.py -hmax 1.28 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.64 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.32 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.16 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.08 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.04 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta 0
python3 example1_thermal_phase_field_fracture.py -hmax 0.01 -alpha_th 1e-5 -theta 0

python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta=240
python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta=160
python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta=80
python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta=-80
python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta=-160
python3 example1_thermal_phase_field_fracture.py -hmax 0.02 -alpha_th 1e-5 -theta=-240
