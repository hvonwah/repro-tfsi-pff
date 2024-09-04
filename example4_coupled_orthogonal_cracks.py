from ngsolve import *
from meshes import *
from netgen.occ import ShapeContinuity, MoveTo, SplineApproximation, OCCGeometry, X, Y, Glue
from phase_field_fracture import *
from fluid_structure_interaction import *
import argparse
import hashlib
from numpy.polynomial.chebyshev import chebfit, chebval, chebroots
import numpy as np

SetNumThreads(4)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=0.5, help='Maximal global mesh size. The initial phase field crack has mesh size smaller by a factor 100.')
options = vars(parser.parse_args())
print(options)

# -------------------------------- PARAMETERS --------------------------------
hmax = options['mesh_size']                     # Global maximal mesh size
hcrack = hmax / 100                             # Crack mesh size
l0 = 0.2                                        # Crack half length

pf_eps = 2                                      # PF regularization * h
pf_gamma = 100                                  # PF penalty parameter * h^-2
kappa = 1e-10                                   # Bulk regularization parameter
G_c = 1                                         # Critical energy release rate
pf_iter = 10                                    # Phase field loading steps
real_compile = True                             # C++ coefficient functions
wait_compile = False                            # Wait for compile to complete
vtk_out = True                                  # Write VTK output

physics_pars = {'E': 1, 'nu_s': 0.3, 'pre': 0, 'pre_0': -4e-2,
                'alpha_B': 0, 'theta_0': 0.0, 'theta': 0.0,
                'alpha_th': 1e-5, 'rho_f': 1, 'nu_f': 0.001,
                'kappa_th': {'fluid': 0.01, 'solid': 1}, 'gravity': 1}

rhs_data_fluid = CF((0.2 * exp(-1000 * ((x - l0)**2 + y**2)), 0))
rhs_data_tmp = 100 * exp(-10 * (x - l0)**2 - 5 * y**4)

hash_pars = hashlib.blake2b(str(physics_pars).encode('utf-8'), digest_size=6).hexdigest()
filename = f'results/example4_orthogonal_cracks_coupled_pff_tfsi_h{hmax}_{hash_pars}'


# ----------------------------- MAIN COMPUTATION -----------------------------
with TaskManager():
    # Make mesh
    mesh = make_T_crack_mesh((-2, -2), (4, 4), l0, l0, hmax, hcrack)
    h = specialcf.mesh_size

    # Step 0: Solve phase-field fracture problem in slit to initialize
    pf_solver = phase_field_crack_solver(
        mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps * h,
        pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
        order=1, real_compile=real_compile, wait_compile=wait_compile,
        formulation='bnd')

    pf_solver.initialize_phase_field()
    pf_solver.solve_phase_field()
    gfu_pff = pf_solver.gfu

    if vtk_out:
        vtk = VTKOutput(
            ma=mesh, coefs=[*gfu_pff.components], names=['def_pf', 'pf'],
            filename=f'{filename}_pff_initial', order=2)
        vtk.Do()

    # Step 1: Crack opening displacement computation
    cod_offset = 20 * hcrack
    n_cod_x = int(2.5 * l0 / hcrack - cod_offset / hcrack)
    lines_x = [-1.5 * l0 + i * hcrack for i in range(n_cod_x)]

    n_cod_y = int(1.5 * l0 / hcrack - cod_offset / hcrack)
    lines_y1 = [-1.5 * l0 + i * hcrack for i in range(n_cod_y)]
    lines_y2 = [cod_offset + i * hcrack for i in reversed(range(n_cod_y))]

    cod_x = cod_from_phase_field(gfu_pff, lines_x)
    cod_y1 = cod_from_phase_field(gfu_pff, lines_y1, True)
    cod_y2 = cod_from_phase_field(gfu_pff, lines_y2, True)

    # Step 2: Geometry reconstruction
    pnts_raw = []
    tips = [1, 1, 1]
    for i, cod in enumerate([cod_x, cod_y1, cod_y2]):
        x, cods = [], []
        for _p in cod:
            if _p[1] > hcrack:
                x.append(_p[0]), cods.append(_p[1])
        x, cods = np.array(x), np.array(cods)
        coefs = chebfit(x, cods, 7)
        fit = chebval(x, coefs)
        roots = [np.real(r) for r in chebroots(coefs) if np.abs(np.imag(r)) < 1e-10]
        roots = [r for r in roots if abs(r - x[0]) < 2 * hcrack]
        if len(roots) == 0:
            tips[i] = 0
            pnts_raw.append([list(x), list(fit)])
        else:
            pnts_raw.append([[roots[0]] + list(x), [0] + list(fit)])

    x, cods = pnts_raw[0]
    bnd_pts = [(p, c / 2) for p, c in zip(reversed(x), reversed(cods))]
    bnd_pts += [(p, - c / 2) for p, c in zip(x[tips[0]:], cods[tips[0]:])]
    x, cods = pnts_raw[1]
    bnd_pts += [(l0 - c / 2, p) for p, c in zip(reversed(x), reversed(cods))]
    bnd_pts += [(l0 + c / 2, p) for p, c in zip(x[tips[1]:], cods[tips[1]:])]
    x, cods = pnts_raw[2]
    bnd_pts += [(l0 + c / 2, p) for p, c in zip(reversed(x), reversed(cods))]
    bnd_pts += [(l0 - c / 2, p) for p, c in zip(x[tips[2]:], cods[tips[2]:])]

    bnd_pts = bnd_pts[-1:] + bnd_pts

    base = MoveTo(-2, -2).Rectangle(4, 4).Face()
    base.faces.name = 'solid'
    base.edges.name = 'out'
    base.faces.maxh = hmax
    fluid = SplineApproximation(bnd_pts, deg_min=2, deg_max=2,
                                continuity=ShapeContinuity(2),
                                tol=1e-8).Face()
    base -= fluid
    fluid.faces.name = 'fluid'
    fluid.faces.maxh = hcrack
    fluid.edges.name = 'interface'

    geo = Glue([base, fluid])
    mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(grading=0.3))

    # Step 3: Solve coupled thermo-fluid-structure interaction problem
    physics_pars['k'] = mesh.MaterialCF(physics_pars['kappa_th'])

    gfu_tfsi = solve_stationary_therm_fsi(
        mesh=mesh, order=2, physics_par=physics_pars,
        rhs_fluid=rhs_data_fluid, rhs_tmp=rhs_data_tmp)

    if vtk_out:
        vtk = VTKOutput(
            ma=mesh, coefs=[*gfu_tfsi.components],
            names=['vel', 'pre', 'def_fsi', 'temp', 'lagr'],
            filename=f'{filename}_tfsi', order=2)
        vtk.Do()

    # Step 4: Solve Phase-Field-Fracture model on new mesh

    # Only couple temperature
    physics_pars['pre'] = gfu_tfsi.components[1]
    pf_solver1 = phase_field_crack_solver(
        mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps * h,
        pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
        order=1, real_compile=real_compile, wait_compile=wait_compile,
        bc_d='out', formulation='bnd')

    pf_solver1.initialize_phase_field()
    pf_solver1.solve_phase_field()
    gfu_u_1, gfu_phi_1 = pf_solver1.gfu.components

    if vtk_out:
        vtk = VTKOutput(
            ma=mesh, coefs=[gfu_u_1, gfu_phi_1], names=['def_pf', 'pf'],
            filename=f'{filename}_pff_coupled_pre', order=2)
        vtk.Do()

    # Only couple temperature and pressure
    physics_pars['theta'] = gfu_tfsi.components[3]
    pf_solver2 = phase_field_crack_solver(
        mesh=mesh, physics_par=physics_pars, pf_eps=pf_eps * h,
        pf_gamma=pf_gamma * h**-2, pf_kappa=kappa, G_c=G_c, n_steps=pf_iter,
        order=1, real_compile=real_compile, wait_compile=wait_compile,
        bc_d='out', formulation='bnd')

    pf_solver2.initialize_phase_field()
    pf_solver2.solve_phase_field()
    gfu_u_2, gfu_phi_2 = pf_solver2.gfu.components

    if vtk_out:
        vtk = VTKOutput(
            ma=mesh, coefs=[gfu_u_2, gfu_phi_2], names=['def_pf', 'pf'],
            filename=f'{filename}_pff_coupled_pre+temp', order=2)
        vtk.Do()
        vtk = VTKOutput(
            ma=mesh, coefs=[gfu_u_2 - gfu_u_1, gfu_phi_2 - gfu_phi_1],
            names=['def_pf', 'pf'], filename=f'{filename}_pff_coupled_diff',
            order=2)
        vtk.Do()
