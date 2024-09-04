from ngsolve import *
from ngsolve.solvers import Newton

__all__ = ['solve_stationary_therm_fsi']


def solve_stationary_therm_fsi(mesh, order, physics_par, rhs_fluid, rhs_tmp, u_d_cf_ih=None, bc_d_ih=None, harmonic_extension=True, alpha_u=1e-14, compile_flag=False, compile_wait=False, newton_info=True, newton_damp=1, newton_tol=1e-11, inverse='pardiso', condense=False, **kwargs):
    '''
    Solve a stationary Navier-Stokes thermo fluid-structure interaction problem.

    Parameters
    ----------
    mesh : ngsolve.Mesh
        Computational mesh for the FSI problem.
    order : int
        Polynomial order of the velocity, deformation and temperature
        spaces.
    physics_par : dict
        Physical parameters for the fluid, solid and temperature models.
    rhs_fluid : ngsolve.CoefficientFunction
        Forcing term for the fluid equation.
    rhs_tmp : ngsolve.CoefficientFunction
        Forcing term for the temperature equation.
    u_d_cf_ih: None or ngsolve.CoefficientFunction
        Inhomogeneous Dirichlet data for the fluid velocity, if
        provided; default=None.
    bc_d_ih: None or str
        Boundary on which the inhomogeneous Dirichlet data is applied;
        default=None.
    harmonic_extension: bool
        Use harmonic (rather than Neohockean) extension to extend
        deformation into fluid region; default=True.
    alpha_u: float
        Harmonic extension parameter; default=1e-14.
    compile_flag: bool
        C++ hard compile integrators; default=False.
    compile_wait: bool
        Wait for hard to compile to complete; default=False.
    newton_info: bool
        Print convergence information from Newton solver; default=True.
    newton_damp: float
        Damping factor in Newton scheme; default=1.
    newton_tol: float
        Tolerance for Newton scheme; default=1e-11.
    inverse: str
        Sparse direct solver to use for linearised systems within Newton
        scheme; default='pardiso'.
    condense: bool
        Use static condensation of higher-order bubbles; default=False.

    Returns
    -------
    ngsolve.GridFunction
        Components: Velocity, pressure, deformation, temperature and
        Pressure Lagrange-multiplayer.
    '''

    for key, value in kwargs.items():
        warnings.warn(f'Unknown kwarg {key}={value}', category=SyntaxWarning)
    for _reg in ['fluid', 'solid']:
        assert _reg in mesh.GetMaterials(), f'The mesh has no "{_reg}" region'
    for _bnd in ['interface', 'out']:
        assert _bnd in mesh.GetBoundaries(), f'The mesh has no "{_bnd}" boundary'

    E = physics_par['E']
    nu_s = physics_par['nu_s']
    mu_s = E / (2 * (1 + nu_s))
    lam_s = E * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))
    rho_f = physics_par['rho_f']
    alpha_th = physics_par['alpha_th']
    k_dr = 2 / 3 * mu_s + lam_s
    theta_0 = physics_par['theta_0']
    nu_f = physics_par['nu_f']
    k = physics_par['k']

    V = VectorH1(mesh, order=order, dirichlet='out')
    Q = H1(mesh, order=order - 1, definedon='fluid')
    D = VectorH1(mesh, order=order, dirichlet='out')
    T = H1(mesh, order=order, dirichlet='out')
    N = NumberSpace(mesh, definedon='fluid')
    X = V * Q * D * T * N
    (u, p, d, th, lam), (v, q, w, ze, mu) = X.TnT()
    Y = V * Q * N
    (_u, _p, _lam), (_v, _q, _mu) = Y.TnT()

    gfu = GridFunction(X)

    Id2 = Id(2)
    F = Grad(d) + Id2
    C = F.trans * F
    E = 0.5 * (Grad(d) + Grad(d).trans)
    J = Det(F)
    Finv = Inv(F)
    FinvT = Finv.trans
    gvec = CF((0, -physics_par['gravity']))

    stress_sol = 2 * mu_s * E + lam_s * Trace(E) * Id2
    stress_sol += -3 * alpha_th * k_dr * (th - theta_0) * Id2
    stress_fl = rho_f * nu_f * (grad(u) * Finv + FinvT * grad(u).trans)
    stress_tmp = k * CF(grad(th), dims=(1, 2))

    diff_fl = InnerProduct(J * stress_fl * FinvT, grad(v))
    conv_fl = rho_f * InnerProduct(Grad(u) * (J * Finv * u), v)
    pres_fl = -J * (Trace(grad(v) * Finv) * p + Trace(grad(u) * Finv) * q)
    pres_fl += - J * lam * q - J * mu * p
    temp_fl = InnerProduct(- J * alpha_th * th * gvec, v)
    rhs_fl = - InnerProduct(rho_f * J * rhs_fluid, v)
    rhs_fl += InnerProduct(J * alpha_th * theta_0 * gvec, v)

    mass_sol = - InnerProduct(u, w)
    el_sol = InnerProduct(J * stress_sol * FinvT, grad(v))

    diff_tmp = InnerProduct(J * stress_tmp * FinvT, grad(ze))
    conv_tmp = InnerProduct(Grad(th) * (J * Finv * u), ze)
    rhs_tmp = -J * rhs_tmp * ze

    if harmonic_extension:
        extension = alpha_u * InnerProduct(Grad(d), Grad(d))
    else:
        gfdist = GridFunction(H1(mesh, order=1, dirichlet=bc_d))
        gfdist.Set(1, definedon=mesh.Boundaries('interface'))

        def NeoHookExt(C, mu=1, lam=1):
            ext = 0.5 * mu * Trace(C - Id2)
            ext += 0.5 * mu * (2 * mu / lam * Det(C)**(-lam / 2 / mu) - 1)
            return ext

        extension = 1 / (1 - gfdist + 1e-2) * 1e-8 * NeoHookExt(C)

    stokes = nu_f * rho_f * InnerProduct(grad(_u), grad(_v))
    stokes += - div(_u) * _q - div(_v) * _p - _lam * _q - _mu * _p

    comp_opt = {'realcompile': compile_flag, 'wait': compile_wait}
    dFL, dSL = dx('fluid', bonus_intorder=order), dx('solid')

    a = BilinearForm(X, symmetric=False, condense=condense)
    a += (diff_fl + conv_fl + pres_fl + temp_fl + rhs_fl).Compile(**comp_opt) * dFL
    a += (mass_sol + el_sol).Compile(**comp_opt) * dSL
    a += Variation(extension.Compile(**comp_opt) * dFL)
    a += (diff_tmp + conv_tmp + rhs_tmp).Compile(**comp_opt) * dx(bonus_intorder=order)

    a_stokes = BilinearForm(Y, symmetric=True, check_unused=False)
    a_stokes += stokes * dFL

    f_stokes = LinearForm(Y)
    f_stokes += InnerProduct(-rhs_fluid, v) * dFL

    bts = Y.FreeDofs() & ~Y.GetDofs(mesh.Materials('solid'))
    bnd_dofs = V.GetDofs(mesh.Boundaries(f'out|interface|{bc_d_ih}'))
    for i in range(V.ndof):
        if bnd_dofs[i]:
            bts[i] = False

    gfu_stokes = GridFunction(Y)
    res_stokes = gfu_stokes.vec.CreateVector()

    a_stokes.Assemble()
    f_stokes.Assemble()
    res_stokes.data = f_stokes.vec

    invstoke = a_stokes.mat.Inverse(bts, inverse=inverse)

    if u_d_cf_ih is not None and bc_d_ih is not None:
        gfu_stokes.components[0].Set(u_d_cf_ih, definedon=mesh.Boundaries(bc_d_ih))
        res_stokes.data -= a_stokes.mat * gfu_stokes.vec

    gfu_stokes.vec.data += invstoke * res_stokes

    gfu.components[0].vec.data = gfu_stokes.components[0].vec
    gfu.components[1].vec.data = gfu_stokes.components[1].vec

    gf_tmp = GridFunction(T)
    gf_tmp.Set(theta_0)
    gfu.components[3].vec.data = gf_tmp.vec

    Newton(a, gfu, maxit=10, inverse=inverse, maxerr=newton_tol,
           dampfactor=newton_damp, printing=newton_info)

    return gfu
