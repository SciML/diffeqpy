from diffeqpy import ode
import pytest


def test_ode_sol_ordinarydiffeq():
    numba = pytest.importorskip('numba')

    def f(u,p,t):
        return -u

    u0 = 0.5
    tspan = (0., 1.)
    prob = ode.ODEProblem(f, u0, tspan)
    sol = ode.solve(prob)
    assert len(sol.t) < 10

    numba_f = numba.jit(f)
    prob = ode.ODEProblem(numba_f, u0, tspan)
    sol2 = ode.solve(prob)
    assert len(sol.t) == len(sol2.t)


def test_lorenz_sol_ordinarydiffeq():
    numba = pytest.importorskip('numba')

    def f(u,p,t):
        x, y, z = u
        sigma, rho, beta = p
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,8/3]
    prob = ode.ODEProblem(f, u0, tspan, p)
    sol = ode.solve(prob)

    def f(du,u,p,t):
        x, y, z = u
        sigma, rho, beta = p
        du[0] = sigma * (y - x)
        du[1] = x * (rho - z) - y
        du[2] = x * y - beta * z

    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,2.66]
    prob = ode.ODEProblem(f, u0, tspan, p)
    sol = ode.solve(prob)

    numba_f = numba.jit(f)
    prob = ode.ODEProblem(numba_f, u0, tspan, p)
    sol2 = ode.solve(prob)
    assert len(sol.t) == len(sol2.t)
