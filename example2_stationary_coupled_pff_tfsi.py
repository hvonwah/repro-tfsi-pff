from ngsolve import *
from meshes import *
from phase_field_fracture import *
from fluid_structure_interaction import *
import argparse
import hashlib
import pandas as pd
import numpy as np

SetNumThreads(4)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=1, help='Maximal global mesh size. The Phase-Field crack has mesh size smaller by a factor 100.')
parser.add_argument('-alpha_th', '--thermal_expansion', type=float, default=1e-5, help='Fracture thermal expansion coefficient')
parser.add_argument('-theta', '--temperature', type=float, default=0, help='Fluid Temperature')
options = vars(parser.parse_args())
print(options)

# -------------------------------- PARAMETERS --------------------------------
hmax = options['mesh_size']                     # Global maximal mesh size
hcrack = hmax / 100                             # Crack mesh size
l0 = 0.2                                        # Crack half length
n_couple_it = 9                                 # Iters. between PFF and TFSI

pf_eps = 2                                      # PF regularisation * h
pf_gamma = 100                                  # PF penalty parameter * h^-2
kappa = 1e-10                                   # Bulk regularization parameter
G_c = 1                                         # Critical energy release rate
pf_iter = 10                                    # Phase field loading steps
real_compile = True                             # C++ coefficient functions
wait_compile = False                            # Wait for compile to complete
vtk_out = True                                  # Write VTK output

physics_pars = {'E': 1, 'nu_s': 0.3, 'pre': 0, 'pre_0': -4e-2,
                'alpha_B': 0, 'theta_0': 0.0, 'theta': options['temperature'],
                'alpha_th': options['thermal_expansion'], 'rho_f': 1,
                'nu_f': 0.001, 'kappa_th': {'fluid': 0.01, 'solid': 1},
                'gravity': 1}

rhs_data_fluid = CF((0.2 * exp(-1000 * ((x - 0)**2 + (y - 0)**2)), 0))
rhs_data_tmp = 100 * exp(-10 * ((x - 0)**2 + (y - 0)**2))

hash_pars = hashlib.blake2b(str(physics_pars).encode('utf-8'), digest_size=6).hexdigest()
filename = f'results/example2_stationary_coupled_pff_tfsi_h{hmax}_{hash_pars}'


# ----------------------------- MAIN COMPUTATION -----------------------------
with TaskManager():
    # Make mesh
    mesh = make_crack_mesh((-2, -2), (4, 4), -l0, l0, hmax, hcrack)
    h = specialcf.mesh_size

    # Solve phase-field fracture problem in slit to initialize
    pf_solver = phase_field_crack_solver(
        mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps * h,
        pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
        order=1, real_compile=real_compile, wait_compile=wait_compile,
        formulation='bnd')

    pf_solver.initialize_phase_field()
    pf_solver.solve_phase_field()

    cod_pnts = [-2 + i * hcrack for i in range(1, int(4 / hcrack))]
    cod_results = pd.DataFrame(cod_pnts, columns=['x'])

    with open(f'{filename}_tcv.dat', 'w') as fid:
        fid.write(f'it tcv\n')

    for it in range(n_couple_it + 1):
        # Step 1: Compute COD (and TCV)
        gfu_pff = pf_solver.gfu
        gfu_u, gfu_phi = gfu_pff.components

        cod = cod_from_phase_field(gfu_pff, cod_pnts)
        cod_results[f'cod{it}'] = np.array(cod)[:, 1]
        cod_results.to_csv(f'{filename}_cod.dat', sep=' ', index=False)

        tcv = tcv_from_phase_field(gfu_pff)
        with open(f'{filename}_tcv.dat', 'a') as fid:
            fid.write(f'{it} {tcv}\n')

        if it == n_couple_it:
            break

        # Step 2: Geometry reconstruction and new mesh
        mesh = make_mesh_from_cod((-2, -2), (4, 4), cod, hmax, hcrack, 0.01)
        h = specialcf.mesh_size

        # Step 3: Solve coupled thermo-fluid-structure interaction problem
        physics_pars['k'] = mesh.MaterialCF(physics_pars['kappa_th'])

        gfu_tfsi = solve_stationary_therm_fsi(
            mesh=mesh, order=2, physics_par=physics_pars,
            rhs_fluid=rhs_data_fluid, rhs_tmp=rhs_data_tmp)
        gfu_vel, gfu_pre, gfu_d, gfu_tmp, gfu_lam = gfu_tfsi.components

        # Step 4: Solve Phase-Field-Fracture model on new mesh
        physics_pars['pre'] = gfu_tfsi.components[1]
        physics_pars['theta'] = gfu_tfsi.components[3]

        pf_solver = phase_field_crack_solver(
            mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps * h,
            pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
            order=1, real_compile=real_compile, wait_compile=wait_compile,
            bc_d='out', formulation='bnd')

        pf_solver.initialize_phase_field()
        pf_solver.solve_phase_field()

        if vtk_out:
            vtk = VTKOutput(
                ma=mesh, coefs=[*gfu_tfsi.components, *gfu_pff.components],
                names=['vel', 'pre', 'def_fsi', 'temp', 'lagr', 'def_pf', 'pf'],
                filename=f'{filename}_step{it + 1}', subdivision=0,
                floatsize='double', legacy=False, order=2)
            vtk.Do()
            vtk = VTKOutput(
                ma=mesh, coefs=[0], names=['zero'],
                filename=f'{filename}_mesh_step{it + 1}', subdivision=0,
                floatsize='single', legacy=False, order=1)
            vtk.Do()
