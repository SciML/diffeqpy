from .. import de
import pytest
numba = pytest.importorskip('numba')


def test():
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
