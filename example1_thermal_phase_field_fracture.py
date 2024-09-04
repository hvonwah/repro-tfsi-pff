from netgen.occ import SplineApproximation, MoveTo, OCCGeometry, Glue, X, Y
from ngsolve import *
from meshes import make_crack_mesh
from phase_field_fracture import *
import argparse
import hashlib

SetNumThreads(4)

parser = argparse.ArgumentParser(description='Compute thermal phase-field fracture problem', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=1, help='Maximal global mesh size. The Phase-Field crack has mesh size smaller by a factor 100.')
parser.add_argument('-alpha_th', '--thermal_expansion', type=float, default=1e-5, help='Fracture thermal expansion coefficient')
parser.add_argument('-theta', '--temperature', type=float, default=0, help='Fluid Temperature')
options = vars(parser.parse_args())
print(options)

# -------------------------------- PARAMETERS --------------------------------
hmax = options['mesh_size']                     # Global maximal mesh size
h_crack = hmax / 100                            # Crack mesh size
l0 = 0.2                                        # Crack half length

pf_eps = 2                                      # PF regularisation * h
pf_gamma = 100                                  # PF penalty parameter * h^-2
kappa = 1e-10                                   # Bulk regularization parameter
G_c = 1                                         # Critical energy release rate
order_pf = 1                                    # Poly. order for phase field
pf_iter = 10                                    # Phase field loading steps
real_compile = True                             # C++ coefficient functions
wait_compile = False                            # Wait for compile to complete
vtk_out = False                                 # Write VTK output
cod_n_pts = 50                                  # Nr. cod values to file

physics_pars = {'E': 1, 'nu_s': 0.3, 'pre': 4e-2, 'pre_0': 0,
                'alpha_B': 0, 'theta_0': 0.0, 'theta': options['temperature'],
                'alpha_th': options['thermal_expansion']}

hash_pars = hashlib.blake2b(str(physics_pars).encode('utf-8'), digest_size=6).hexdigest()
filename = f'results/example1_thermal_phase_field_fracture_h{hmax}_{hash_pars}'


# ----------------------------- MAIN COMPUTATION -----------------------------
with TaskManager():
    # Make mesh
    mesh = make_crack_mesh((-2, -2), (4, 4), -l0, l0, hmax, h_crack)
    h = specialcf.mesh_size

    # Solve phase-field fracture problem
    pf_solver = phase_field_crack_solver(
        mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps * h,
        pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
        order=order_pf, real_compile=real_compile, wait_compile=wait_compile,
        formulation='bnd')

    pf_solver.initialize_phase_field()
    pf_solver.solve_phase_field()

    # Compute quantities of interest
    gfu = pf_solver.gfu
    gfu_u, gfu_phi = gfu.components

    # Compute CODs
    cod_pnts = [-l0 * 1.25 + 2.5 * l0 * i / cod_n_pts for i in range(cod_n_pts + 1)]
    cod = cod_from_phase_field(gfu, cod_pnts)
    with open(f'{filename}_cod.dat', 'w') as fid:
        fid.write('x cod\n')
        for p, c in cod:
            fid.write(f'{p} {c}\n')

    # Compute TCV
    tcv = tcv_from_phase_field(gfu)
    with open(f'{filename}_tcv.dat', 'w') as fid:
        fid.write(f'{tcv}')


# Write VTK output
if vtk_out:
    vtk = VTKOutput(gfu_u.space.mesh, [gfu_u, gfu_phi], ['deform', 'phi'],
                    filename, subdivision=1, order=1)
    vtk.Do()
