import sys
from . import load_julia_packages, _declared_julia_packages
# Match the package set declared in diffeqpy/juliapkg.json. DifferentialEquations.jl
# v8 only re-exports SciMLBase + OrdinaryDiffEq, and OrdinaryDiffEq v7 only
# exports the default solvers, so the ODE solver sublibraries and the
# SDE/DDE/DAE/callback packages (and their exported symbols, e.g. RadauIIA5,
# MethodOfSteps, IDA, SOSRI) are not accessible via `de.<name>` if we just take a
# reference to DifferentialEquations. Load the full stack and build a single
# merged module that `using`s each package, so `de.<name>` resolves uniformly
# for everything `from diffeqpy import de` users used to get pre-v8.
solver_packages = _declared_julia_packages()
load_julia_packages(*solver_packages)
from juliacall import Main
de = Main.seval(
    "module _diffeqpy_de; using "
    + ", ".join(solver_packages)
    + "; end; _diffeqpy_de"
)
de.seval("jit(x) = typeof(x).name.wrapper(ModelingToolkit.complete(ModelingToolkit.modelingtoolkitize(x); split = false), [], x.tspan)") # kinda hackey
de.seval("""
                      function jit(x)
                          prob = typeof(x).name.wrapper(ModelingToolkit.complete(ModelingToolkit.modelingtoolkitize(x); split = false), [], Float32.(x.tspan))
                          ModelingToolkit.remake(prob; u0 = Float32.(prob.u0), p = Float32.(prob.p))
                      end
                      """) # kinda hackey
sys.modules[__name__] = de # mutate myself
