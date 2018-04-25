try:
    import numba
except ImportError:
    HAVE_NUMBA=False
else:
    HAVE_NUMBA=True

import pytest
import diffeqpy
de = diffeqpy.setup()

def f1(u,p,t):
    return -u


def test_ode_sol():
    u0 = 0.5
    tspan = (0., 1.)
    prob = de.ODEProblem(f1, u0, tspan)
    sol = de.pysolve(prob)
    assert len(sol.t) < 10


@pytest.mark.skipif(not HAVE_NUMBA, "The package 'numba' is needed for this test")
def test_ode_sol_numba():
    numba_f1 = numba.jit(f1)
    prob = de.ODEProblem(numba_f1, u0, tspan)
    sol2 = de.pysolve(prob)
    assert len(sol.t) == len(sol2.t)


def f_lorenz(u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def test_lorenz_sol():
    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,8/3]
    prob = de.ODEProblem(f_lorenz, u0, tspan, p)
    sol = de.pysolve(prob)

def f_lorenz2(du,u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    du[0] = sigma * (y - x)
    du[1] = x * (rho - z) - y
    du[2] = x * y - beta * z


def test_lorenz2_sol():
    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,2.66]
    prob = de.ODEProblem(f_lorenz2, u0, tspan, p)
    sol = de.pysolve(prob)


@pytest.mark.skipif(not HAVE_NUMBA, "The package 'numba' is needed for this test")
def test_lorenz2_sol_numba():
    numba_f = numba.jit(f_lorenz2)
    prob = de.ODEProblem(numba_f, u0, tspan, p)
    sol2 = de.pysolve(prob)
