from .. import de


def test():
    f = de.seval("""
    function f(du, u, h, p, t)
      du[1] = 1.1/(1 + sqrt(10)*(h(p, t-20)[1])^(5/4)) - 10*u[1]/(1 + 40*u[2])
      du[2] = 100*u[1]/(1 + 40*u[2]) - 2.43*u[2]
    end""")
    u0 = [1.05767027/3, 1.030713491/3]

    h = de.seval("""
    function h(p,t)
      [1.05767027/3, 1.030713491/3]
    end""")

    tspan = (0.0, 100.0)
    constant_lags = [20.0]
    prob = de.DDEProblem(f,u0,h,tspan,constant_lags=constant_lags)
    sol = de.solve(prob,saveat=0.1)
