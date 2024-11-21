from .. import de
import pytest


def test():
    numba = pytest.importorskip('numba')

    def f(u,p,t):
        return 1.01*u

    def g(u,p,t):
        return 0.87*u

    u0 = 0.5
    tspan = (0.0,1.0)
    prob = de.SDEProblem(f,g,u0,tspan)
    sol = de.solve(prob)

    def f(du,u,p,t):
        x, y, z = u
        sigma, rho, beta = p
        du[0] = sigma * (y - x)
        du[1] = x * (rho - z) - y
        du[2] = x * y - beta * z

    def g(du,u,p,t):
        du[0] = 0.3*u[0]
        du[1] = 0.3*u[1]
        du[2] = 0.3*u[2]

    numba_f = numba.jit(f)
    numba_g = numba.jit(g)
    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,2.66]
    prob = de.SDEProblem(numba_f, numba_g, u0, tspan, p)
    sol = de.solve(prob)


def test_jit():

    def f(du, u, p, t):
        x, y, z = u
        sigma, rho, beta = p
        du[0] = sigma * (y - x)
        du[1] = x * (rho - z) - y
        du[2] = x * y - beta * z

    def g(du, u, p, t):
        du[0] = 0.03 * u[0]
        du[1] = 0.03 * u[1]
        du[2] = 0.03 * u[2]

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    p = [10.0, 28.0, 2.66]
    prob = de.jit(de.SDEProblem(f, g, u0, tspan, p))
    sol = de.solve(prob)
    assert sol.t[-1] == tspan[-1], f"Solver did not reach the final time. Last time: {sol.t[-1]}"
    assert len(sol.u) > 0, "Solution is empty."
    assert all(
        abs(sol.u[i][j]) < float("inf") for j in range(len(u0)) for i in range(len(sol.t))
    ), "Solution contains non-finite values."