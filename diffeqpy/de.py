import sys
from . import load_julia_packages
# Match the package set declared in diffeqpy/juliapkg.json. DifferentialEquations.jl
# v8 only re-exports SciMLBase + OrdinaryDiffEq, so the SDE/DDE/DAE/callback
# sub-solver packages (and their exported symbols, e.g. MethodOfSteps, IDA,
# SOSRI) are no longer accessible via `de.<name>` if we just take a reference
# to DifferentialEquations. Load the full stack and build a single merged
# module that `using`s each sublib, so `de.<name>` resolves uniformly for
# everything `from diffeqpy import de` users used to get pre-v8.
load_julia_packages(
    "DifferentialEquations",
    "OrdinaryDiffEq",
    "OrdinaryDiffEqDefault",
    "StochasticDiffEq",
    "DelayDiffEq",
    "Sundials",
    "DiffEqCallbacks",
    "ModelingToolkit",
)
from juliacall import Main
de = Main.seval("""
    module _diffeqpy_de
        using DifferentialEquations
        using OrdinaryDiffEq, OrdinaryDiffEqDefault, StochasticDiffEq
        using DelayDiffEq, Sundials, DiffEqCallbacks, ModelingToolkit
    end
    _diffeqpy_de
""")
de.seval("jit(x) = typeof(x).name.wrapper(ModelingToolkit.complete(ModelingToolkit.modelingtoolkitize(x); split = false), [], x.tspan)") # kinda hackey
de.seval("""
                      function jit(x)
                          prob = typeof(x).name.wrapper(ModelingToolkit.complete(ModelingToolkit.modelingtoolkitize(x); split = false), [], Float32.(x.tspan))
                          ModelingToolkit.remake(prob; u0 = Float32.(prob.u0), p = Float32.(prob.p))
                      end
                      """) # kinda hackey
sys.modules[__name__] = de # mutate myself
