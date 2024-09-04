from netgen.occ import SplineApproximation, MoveTo, OCCGeometry, Glue, X, Y
from ngsolve import Mesh
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, chebroots
from math import floor

__all__ = ['make_crack_mesh', 'make_mesh_from_cod', 'make_T_crack_mesh']


def make_crack_mesh(corner, rectangle, l0, l1, hmax, hcrack):
    base = MoveTo(*corner).Rectangle(*rectangle).Face()
    crack = MoveTo(l0, -hcrack)
    crack = crack.Rectangle(abs(l0) + l1, 2 * hcrack).Face()
    base -= crack

    base.faces.name = 'solid'
    base.faces.maxh = hmax
    base.edges.Min(Y).name = 'bottom'
    base.edges.Max(X).name = 'right'
    base.edges.Max(Y).name = 'top'
    base.edges.Min(X).name = 'left'
    crack.faces.name = 'crack'
    crack.faces.maxh = hcrack
    crack.faces.col = (1, 0, 0)
    crack.edges.name = 'interface'

    geo = Glue([base, crack])
    mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(grading=0.2))
    return mesh


def make_T_crack_mesh(corner, rectangle, l0, l1, hmax, hcrack):
    base = MoveTo(*corner).Rectangle(*rectangle).Face()
    crack = MoveTo(-l0, - hcrack).Line(2 * l0 - hcrack, 0).Line(0, -l1 + hcrack)
    crack = crack.Line(2 * hcrack, 0).Line(0, 2 * l1).Line(-2 * hcrack, 0)
    crack = crack.Line(0, -l1 + hcrack).Line(-2 * l0 + hcrack, 0).Close().Face()
    base -= crack

    base.faces.name = 'solid'
    base.faces.maxh = hmax
    base.edges.Min(Y).name = 'bottom'
    base.edges.Max(X).name = 'right'
    base.edges.Max(Y).name = 'top'
    base.edges.Min(X).name = 'left'
    crack.faces.name = 'crack'
    crack.faces.maxh = hcrack
    crack.faces.col = (1, 0, 0)
    crack.edges.name = 'interface'

    geo = Glue([base, crack])
    mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(grading=0.3))
    return mesh


def make_mesh_from_cod(corner, rectangle, cod_data, hmax, hcrack, curvaturesafety, extrapol_tips=False):
    base = MoveTo(*corner).Rectangle(*rectangle).Face()

    for _i in range(floor(len(cod_data) / 2)):
        if cod_data[_i][1] < hcrack / 4:
            i1 = _i + 1
        if cod_data[-_i - 1][1] < hcrack / 4:
            i2 = -_i - 1
    cod_data = cod_data[i1: i2]

    # Pre-process COD data
    if extrapol_tips:
        cod_data = np.array(cod_data)
        x, cod = cod_data[:, 0], cod_data[:, 1]

        coefs = chebfit(x, cod, 13)
        fit = chebval(x, coefs) / 2

        # Get inner most real roots
        roots = [np.real(r) for r in chebroots(coefs) if np.abs(np.imag(r)) < 1e-10]
        for i in range(len(roots) - 1):
            if np.sign(roots[i]) != np.sign(roots[i + 1]):
                tips = roots[i], roots[i + 1]
                pts = np.append(np.dstack((x, fit))[0], [[tips[1], 0]], axis=0)
                pts = np.append(pts, np.dstack((x[::-1], -fit[::-1]))[0], axis=0)
                pts = np.append(pts, [[tips[0], 0]], axis=0)
                break
        else:
            print('Failed to find crack tipts')
            pts = np.append(np.dstack((x, fit))[0],
                            np.dstack((x[::-1], -fit[::-1]))[0], axis=0)
        pts = np.append(pts, [pts[0, :]], axis=0)
        pts = [tuple(_p) for _p in np.flip(pts, axis=0)]
    else:
        pts = [(p, c / 2) for (p, c) in cod_data]
        pts += [(p, -c / 2) for (p, c) in reversed(cod_data)] + [pts[0]]
        pts.reverse()

    fluid = SplineApproximation(pts, deg_min=1, deg_max=1).Face()
    fluid *= base
    base -= fluid

    base.edges.name = 'out'
    base.faces.name = 'solid'
    base.faces.maxh = hmax
    fluid.faces.name = 'fluid'
    fluid.faces.maxh = hcrack
    fluid.faces.col = (1, 0, 0)
    fluid.edges.name = 'interface'

    geo = OCCGeometry(Glue([base, fluid]), dim=2)
    ngmesh = geo.GenerateMesh(grading=0.3, curvaturesafety=curvaturesafety)
    mesh = Mesh(ngmesh)

    if extrapol_tips is True:
        return mesh, tips
    else:
        return mesh
