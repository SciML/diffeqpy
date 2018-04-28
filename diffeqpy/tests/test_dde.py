from .. import de


def test():
    def f(du,u,p,t):
        resid1 = - 0.04*u[0]               + 1e4*u[1]*u[2] - du[0]
        resid2 = + 0.04*u[0] - 3e7*u[1]**2 - 1e4*u[1]*u[2] - du[1]
        resid3 = u[0] + u[1] + u[2] - 1.0
        return [resid1,resid2,resid3]

    u0 = [1.0, 0, 0]
    du0 = [-0.04, 0.04, 0.0]
    tspan = (0.0,100000.0)
    differential_vars = [True,True,False]
    prob = de.DAEProblem(f,du0,u0,tspan,differential_vars=differential_vars)
    sol = de.solve(prob)
