import sys
from . import load_julia_packages
# `DifferentialEquations.jl` v8 only re-exports SciMLBase + OrdinaryDiffEq, so the
# SDE/DDE/DAE/callback sub-solver packages have to be loaded explicitly here for
# their __init__ hooks to register the default algorithms used by `de.solve(prob)`.
# OrdinaryDiffEqDefault is pulled in transitively by OrdinaryDiffEq and now also
# supplies the default DAEProblem solver.
loaded = load_julia_packages(
    "DifferentialEquations",
    "StochasticDiffEq",
    "DelayDiffEq",
    "Sundials",
    "DiffEqCallbacks",
    "ModelingToolkit",
    "PythonCall",
)
de = loaded[0]
from juliacall import Main
de.seval("jit(x) = typeof(x).name.wrapper(Main.ModelingToolkit.complete(Main.ModelingToolkit.modelingtoolkitize(x); split = false), [], x.tspan)") # kinda hackey
de.seval("""
                      function jit(x)
                          prob = typeof(x).name.wrapper(Main.ModelingToolkit.complete(Main.ModelingToolkit.modelingtoolkitize(x); split = false), [], Float32.(x.tspan))
                          Main.ModelingToolkit.remake(prob; u0 = Float32.(prob.u0), p = Float32.(prob.p))
                      end
                      """) # kinda hackey
sys.modules[__name__] = de # mutate myself
