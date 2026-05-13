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
    # Pass an explicit algorithm rather than relying on the default selector.
    # The `MethodOfSteps(DefaultODEAlgorithm())` path currently hits a
    # `Cannot convert Rosenbrock23Cache{...}` MethodError from a ForwardDiff
    # Tag asymmetry between DDE and ODE default-alg selection in
    # SciML/OrdinaryDiffEq.jl — see diffeqpy #172 for the upstream tracking.
    sol = de.solve(prob,de.MethodOfSteps(de.Tsit5()),saveat=0.1)
