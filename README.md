[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13685486.svg)](https://doi.org/10.5281/zenodo.13685486)

This repository contains the reproduction scripts and resulting data for:"S. Lee, H. v. Wahl and T. Wick, A thermo-flow-mechanics-fracture model coupling a phase-field interface approach and thermo-fluid-structure interaction.  [arXiv:2409.03416](https://arxiv.org/abs/2409.03416) [math.NA]". The code is based in parts on our previous work [1,2].

**References**

[1] H. von Wahl and T. Wick. A high-accuracy framework for phase-field fracture interface reconstructions with application to Stokes fluid-filled fracture surrounded by an elastic medium. Comput. Methods Appl. Mech. Engrg., 415:116202, October 2023, [doi:10.1016/j.cma.2023.116202](https://doi.org/10.1016/j.cma.2023.116202). [Code repository](https://github.com/hvonwah/stationary_phase_field_stokes_fsi).

[2] H. von Wahl, T. Wick, A coupled high-accuracy phase-field fluid–structure interaction framework for stokes fluid-filled fracture surrounded by an elastic medium, Results Appl. Math. 22:100455, May 2024, [doi:10.1016/j.rinam.2024.100455](https://doi.org/10.1016/j.rinam.2024.100455). [Code repository](https://github.com/hvonwah/repro-coupled-phase-field-fsi).


# Files
```
+- README.md                                   // This file
+- LICENSE                                     // The license file
+- install.txt                                 // Installation help
+- example1_thermal_phase_field_fracture.py    // Implementation of Example 1
+- example2_stationary_coupled_pff_tfsi.py     // Implementation of Example 2
+- example3a_propagating_pff_reconstr.py       // Implementation of Example 3a
+- example3b_propagating_pff_reconstr.py       // Implementation of Example 3b
+- example4_coupled_orthogonal_cracks.py       // Implementation of Example 4
+- fluid_structure_interaction.py              // TFSI using ALE implementation
+- meshes.py                                   // Mesh construction functions
+- phase_field_fracture.py                     // Interface phase-field fracture implementation
+- postprocess.py                              // Post-process convergence results for example 1
+- run_example1.bash                           // Run parameter studies as in paper
+- run_example2.bash                           // Run parameter studies as in paper
+- run_example3a.bash                          // Run parameter studies as in paper
+- run_example3b.bash                          // Run parameter studies as in paper
+- run_example4.bash                           // Run parameter studies as in paper
+- results/*                                   // The raw text files produced by the computations 
```

# Installation

See the instructions in `install.txt`

# How to reproduce
The scripts to reproduce the computational results are located in the base folder. The resulting data is located in the `results` directory.

The individual examples are implemented the `example_*.py` scripts. These can be executed with the parameters as presented in the manuscript using the bash scripts `run_example*.bash.`

By default, the direct solver `pardiso` is used to solve the linear systems resulting from the discretisation. If this is not available, this may be replaced with `umfpack` in the `DATA` block of each example python study script.
