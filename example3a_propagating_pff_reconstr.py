from ngsolve import *
from meshes import *
from phase_field_fracture import *
import argparse
import hashlib
import pandas as pd
import numpy as np

SetNumThreads(4)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=0.5, help='Maximal global mesh size. The Phase-Field crack has mesh size smaller by a factor 100.')
parser.add_argument('-eps', '--pf_eps', type=float, default=0.01, help='PF Epsilon')
options = vars(parser.parse_args())
print(options)

# -------------------------------- PARAMETERS --------------------------------
hmax = options['mesh_size']                     # Global maximal mesh size
hcrack = hmax / 100                             # Crack mesh size
l0 = 0.2                                        # Crack half length
n_steps = 100                                   # Number of pseudo time-steps

pf_eps = options['pf_eps']                      # PF regularization
pf_gamma = 100                                  # PF penalty parameter * h^-2
kappa = 1e-10                                   # Bulk regularization parameter
G_c = 1                                         # Critical energy release rate
pf_iter = 10                                    # Phase field loading steps
real_compile = True                             # C++ coefficient functions
wait_compile = False                            # Wait for compile to complete
vtk_out = True                                  # Write VTK output

physics_pars = {'E': 1, 'nu_s': 0.3, 'pre': 4e-2, 'pre_0': 0.0, 'alpha_B': 0,
                'theta_0': 0.0, 'theta': 0.0, 'alpha_th': 0.0, 'rho_f': 1,
                'nu_f': 0.001, 'kappa_th': {'fluid': 0.01, 'solid': 1},
                'gravity': 1}

hash_pars = hashlib.blake2b(str(physics_pars).encode('utf-8'), digest_size=6).hexdigest()
filename = f'results/example3a_propagating_pff_h{hmax}eps{pf_eps}_{hash_pars}'


def p_func(n):
    return 4e-2 + n * 1e-4


# ----------------------------- MAIN COMPUTATION -----------------------------
with open(f'{filename}.dat', 'w') as fid:
    fid.write('step tcv left right\n')


with TaskManager():
    # Initialization
    mesh = make_crack_mesh((-2, -2), (4, 4), -l0, l0, hmax, hcrack)
    h = specialcf.mesh_size

    # Solve phase-field fracture problem in slit to initialize
    pf_solver = phase_field_crack_solver(
        mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps,
        pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
        order=1, real_compile=real_compile, wait_compile=wait_compile,
        formulation='bnd')

    pf_solver.initialize_phase_field()
    pf_solver.solve_phase_field()

    if vtk_out:
        vtk = VTKOutput(
            ma=mesh, coefs=[*pf_solver.gfu.components], names=['def_pf', 'pf'],
            filename=f'{filename}_step0', subdivision=0,
            floatsize='double', legacy=False, order=2)
        vtk.Do()
        vtk = VTKOutput(
            ma=mesh, coefs=[0], names=['zero'],
            filename=f'{filename}_mesh_step0', subdivision=0,
            floatsize='single', legacy=False, order=1)
        vtk.Do()

    cod_pnts = [-2 + i * hcrack for i in range(1, int(4 / hcrack))]
    cod_results = {'x': np.array(cod_pnts)}

    for it in range(n_steps):
        # Step 1: Compute COD (and TCV)
        cod = cod_from_phase_field(pf_solver.gfu, cod_pnts)
        tcv = tcv_from_phase_field(pf_solver.gfu)

        cod_results[f'cod{it}'] = np.array(cod)[:, 1]
        df = pd.DataFrame(cod_results)
        df.to_csv(f'{filename}_cod.dat', sep=' ', index=False)

        # Step 2: Geometry reconstruction and new mesh
        mesh, tips = make_mesh_from_cod((-2, -2), (4, 4), cod, hmax,
                                        hcrack, 0.01, True)

        with open(f'{filename}.dat', 'a') as fid:
            fid.write(f'{it} {tcv} {tips[0]} {tips[1]}\n')
        if it == n_steps - 1:
            break

        # Step 3:
        physics_pars['pre'] = p_func(it)

        # Step 4: Solve Phase-Field-Fracture model on new mesh
        pf_solver = phase_field_crack_solver(
            mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps,
            pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c,
            n_steps=pf_iter, order=1, real_compile=real_compile,
            wait_compile=wait_compile, bc_d='out', formulation='bnd')

        pf_solver.initialize_phase_field()
        pf_solver.solve_phase_field()

        if vtk_out:
            vtk = VTKOutput(
                ma=mesh, coefs=[*pf_solver.gfu.components], names=['def_pf', 'pf'],
                filename=f'{filename}_step{it + 1}', subdivision=0,
                floatsize='double', legacy=False, order=2)
            vtk.Do()
            vtk = VTKOutput(
                ma=mesh, coefs=[0], names=['zero'],
                filename=f'{filename}_mesh_step{it + 1}', subdivision=0,
                floatsize='single', legacy=False, order=1)
            vtk.Do()
