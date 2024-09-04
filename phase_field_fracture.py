from ngsolve import *
from ngsolve.solvers import Newton
from xfem import *

__all__ = ['phase_field_crack_solver', 'cod_from_phase_field', 'tcv_from_phase_field']


def pos(x):
    return IfPos(x, x, 0)


class phase_field_crack_solver():
    '''
    Class containing the tools and steps to compute a phase-field
    fracture model.

    Attributes
    ----------
    mesh : ngsolve.Mesh
        Computational mesh.
    pre : float or ngsolve.CoefficientFuntion
        Pressure driving the fracture.
    theta: float or ngsolve.CoefficientFunction
        Temperature of the material.
    eps : float
        Phase-field regularization parameter.
    gamma : float
        Penalty parameter.
    kappa : float
        Bulk regularisation parameter.
    n_steps : integer
        Number of loading steps.
    info : boolean
        Print info during phase-field computation
    gfu : ngsolve.GridFuntion
        Finite element function for the phase-field and displacement
        solutions.
    gfu_last : ngsolve.GridFuntion
        Store the solution from the last (loading) step
    a : ngsolve.BilinearForm
        Weak form of the phase-field problem PDE.
    inverse : string
        Sparse direct solver for linear systems

    Methods
    -------
    initialize_phase_field
        Initialize the phase-field solution according to the mesh
        materials.

    solve_phase_field
        Solve the Phase-field problem with given initial data.
    '''

    def __init__(self, mesh, physics_par, pf_eps, pf_gamma, pf_kappa=1e-10, G_c=5e2, n_steps=10, order=1, bc_d='bottom|right|top|left', formulation='bnd', inverse='pardiso', real_compile=True, wait_compile=True, info=True, **kwargs):
        '''
        Set up phase-field solver.

        Parameters
        ----------
        mesh : ngsolve.Mesh
            Computational mesh
        physics_par : dict
            Physics parameters.
        pf_eps : float
            Phase-field regularisation parameter.
        pf_gamma : float
            Phase field penalisation parameter.
        pf_kappa : float
            Phase field bulk regularisation parameter; default=1e-10.
        G_c : float
            Critical energy release rate; default=5e2.
        n_steps : int
            Number of phase-field iterations; default=10
        order : int
            Polynomial order of the FE space; default=1.
        bc_d : string
            Expression for the Dirichlet boundary condition for the phase-field
            deformation; default='bottom|right|top|left'.
        formulation : str
            'vol' to use volume pressure and temperature coupling formulation
            or 'int' for use interface pressure and temperature coupling.
        inverse : string
            Sparse direct solver to for linear systems; default='pardiso'.
        real_compile : bool
            C++ hard compile integrators; default=True.
        wait_compile : bool
            Wait for hard compile to complete; default=True.
        info : bool
            Print information to terminal; default=True.
        '''

        for key, value in kwargs.items():
            warnings.warn(f'Unknown keyword argument {key}={value}', category=SyntaxWarning)

        self.eps = pf_eps
        self.gamma = pf_gamma
        self.kappa = pf_kappa
        self.G_c = G_c
        self.n_steps = n_steps
        self.info = info

        self.mesh = mesh
        self.order = order
        self.inverse = inverse
        self.compile_opts = {'realcompile': real_compile, 'wait': wait_compile}

        self.pre = physics_par['pre']
        self.theta = physics_par['theta']
        E = physics_par['E']
        nu_s = physics_par['nu_s']
        mu_s = E / (2 * (1 + nu_s))
        lam_s = nu_s * E / ((1 + nu_s) * (1 - 2 * nu_s))
        k_dr = 2 / 3 * mu_s + lam_s
        alpha_B = physics_par['alpha_B']
        pre_0 = physics_par['pre_0']
        alpha_th = physics_par['alpha_th']
        theta_0 = physics_par['theta_0']

        V1 = VectorH1(self.mesh, order=order, dirichlet=bc_d)
        V2 = H1(self.mesh, order=order)
        V = V1 * V2
        if self.info is True:
            print(f'Nr FreeDofs of PF space = {sum(V.FreeDofs())}')

        def e(U):
            return 1 / 2 * (Grad(U) + Grad(U).trans)

        def sigma_s(U, sigma_R=False):
            Id_mat = Id(self.mesh.dim)
            sigma_s = 2 * mu_s * e(U) + lam_s * Trace(e(U)) * Id_mat
            # if sigma_R is True:
            #     sigma_s += - alpha_B * (self.pre - pre_0) * Id_mat
            #     sigma_s += - 3 * alpha_th * k_dr * (self.theta - theta_0) * Id_mat
            return sigma_s

        (u, phi), (v, psi) = V.TnT()

        self.gfu = GridFunction(V)
        self.gfu_last = GridFunction(V)

        phi_last = self.gfu_last.components[1]
        n = specialcf.normal(self.mesh.dim)

        g_phi_last = ((1 - self.kappa) * phi_last**2 + self.kappa)

        form = g_phi_last * InnerProduct(sigma_s(u), e(v))
        form += (1 - self.kappa) * phi * InnerProduct(sigma_s(u), e(u)) * psi
        form += self.G_c * self.eps * InnerProduct(Grad(phi), Grad(psi))
        form += self.G_c * (- 1 / self.eps) * (1 - phi) * psi
        form += self.gamma * pos(phi - phi_last) * psi

        if formulation == 'vol':
            form += g_phi_last * (self.pre - pre_0) * div(v)
            form += g_phi_last * InnerProduct(Grad(self.pre) - Grad(pre_0), v)
            form += alpha_th * g_phi_last * (self.theta - theta_0) * div(v)
            form += alpha_th * g_phi_last * InnerProduct(Grad(self.theta) - Grad(theta_0), v)
            form += 2 * (1 - self.kappa) * phi * (self.pre - pre_0) * div(u) * psi
            form += 2 * (1 - self.kappa) * phi * InnerProduct(Grad(self.pre) - Grad(pre_0), u) * psi
            form += 2 * (1 - self.kappa) * alpha_th * phi * (self.theta - theta_0) * div(u) * psi
            form += 2 * (1 - self.kappa) * alpha_th * phi * InnerProduct(Grad(self.theta) - Grad(theta_0), u) * psi
        elif formulation == 'bnd':
            form_bnd = - (1 - alpha_B) * (self.pre - pre_0) * n * v
            form_bnd += - (alpha_th - 3 * alpha_th * k_dr) * (self.theta - theta_0) * n * v
            form_bnd += - 2 * (1 - self.kappa) * (1 - alpha_B) * (self.pre - pre_0) * n * u * psi
            form_bnd += - 2 * (1 - self.kappa) * (alpha_th - 3 * alpha_th * k_dr) * (self.theta - theta_0) * n * u * psi
        else:
            raise ValueError('Dont knwo which formulation to use')

        dx_interface = dx(definedon=self.mesh.Boundaries('interface'))

        self.a = BilinearForm(V, symmetric=False)
        self.a += form.Compile(**self.compile_opts) * dx

        if formulation == 'bnd':
            self.a += form_bnd.Compile(**self.compile_opts) * dx_interface

        return None

    def initialize_phase_field(self):
        '''
        Initialize phase field with 0 in crack an 1 in solid.
        '''
        if self.info is True:
            print('Initialzie Phase-Field')

        V_loc = self.gfu.components[1].space
        freedofs = V_loc.FreeDofs()
        gf_phi, phi_last = GridFunction(V_loc), GridFunction(V_loc)

        gf_phi_inv = GridFunction(V_loc)
        gf_phi_inv.Set(1, definedon=self.mesh.Materials('crack|fluid'))
        gf_phi.Set(1 - gf_phi_inv)

        phi, psi = V_loc.TnT()
        form = self.G_c * self.eps * InnerProduct(Grad(phi), Grad(psi))
        form += self.G_c * (- 1 / self.eps) * (1 - phi) * psi
        form += self.gamma * pos(phi - phi_last) * psi

        a_loc = BilinearForm(V_loc)
        a_loc += form.Compile() * dx

        for i in range(self.n_steps):
            phi_last.vec.data = gf_phi.vec
            out = Newton(a_loc, gf_phi, freedofs=freedofs,
                         inverse=self.inverse, printing=False, maxerr=1e-8)
            if self.info is True:
                update = Norm(phi_last.vec - gf_phi.vec) / sum(freedofs)
                print(f'{i}, {out[1]:2d}, {update:.3e}')

        self.gfu.vec.data[:] = 0.0
        self.gfu.components[1].vec.data[:] = gf_phi.vec

        del a_loc, V_loc, gf_phi, phi_last, gf_phi_inv

        return None

    def solve_phase_field(self, u_d=None, dirichlet=None):
        '''
        Solve the phase-field problem with initial state stored in
        self.gfu.

        u_d : ngsolve.CoefficientFunktion
            If provided, the Dirichlet boundary condition for the deformation.
        dirichlet : string
            The names of the boundaries where the inhomogeneous Dirichlet
            boundary condition is to be applied.

        Returns
        -------
        self.gfu : ngsolve.GridFunction
            Finite element solution to phase-field problem.
        '''
        if self.info is True:
            print('Solve Phase-Field')

        if ((u_d is not None and dirichlet is None)
                or (u_d is None and dirichlet is not None)):
            raise ValueError('Both u_d and dirichlet have to be provided')
        if u_d is not None:
            gfu_u, gfu_pf = self.gfu_last.components
            gfu_u.Set(u_d, definedon=self.mesh.Boundaries(dirichlet))
            self.gfu.components[0].vec.data = gfu_u.vec

        freedofs = self.gfu.space.FreeDofs()

        for i in range(self.n_steps):
            self.gfu_last.vec.data = self.gfu.vec
            out = Newton(self.a, self.gfu, freedofs=freedofs,
                         inverse=self.inverse, printing=False, maxerr=1e-8)
            if self.info is True:
                update = Norm(self.gfu_last.vec - self.gfu.vec) / sum(freedofs)
                print(f'{i}, {out[1]:2d}, {update:.3e}')
        return self.gfu


def cod_from_phase_field(gfu, lines, vertical=False):
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    order = gf_u.space.globalorder

    _x = CF(x)
    if vertical is True:
        _x = CF(y)

    # Compute crack opening width based on phase-field
    lsetp1_line = GridFunction(H1(mesh, order=1))
    InterpolateToP1(_x - 2, lsetp1_line)
    ci_line = CutInfo(mesh, lsetp1_line)
    el_line = ci_line.GetElementsOfType(IF)
    ds_line = dCut(lsetp1_line, IF, order=2 * order, definedonelements=el_line)

    line_ind = InnerProduct(gf_u, Grad(gf_phi)).Compile()

    crack_openings = []
    for x0 in lines:
        InterpolateToP1(_x - x0, lsetp1_line)
        ci_line.Update(lsetp1_line)
        _cod = Integrate(line_ind * ds_line, mesh)

        crack_openings.append((x0, _cod))

    return crack_openings


def tcv_from_phase_field(gfu):
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    order = gf_u.space.globalorder
    return Integrate(InnerProduct(gf_u, Grad(gf_phi)).Compile(), mesh,
                     order=2 * order - 1)
