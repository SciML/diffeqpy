import sys
from . import load_julia_packages, _ode_solver_packages
# Match the OrdinaryDiffEq sublibrary set declared in diffeqpy/juliapkg.json.
# OrdinaryDiffEq v7 only exports the default solvers (Tsit5, the Verns,
# Rosenbrock23, Rodas5P, FBDF); every other algorithm lives in a solver
# sublibrary. Build a merged module that `using`s each of them so `ode.<name>`
# resolves for the full solver set, as it did with OrdinaryDiffEq v6.
solver_packages = _ode_solver_packages()
load_julia_packages(*solver_packages)
from juliacall import Main
ode = Main.seval(
    "module _diffeqpy_ode; using "
    + ", ".join(solver_packages)
    + "; end; _diffeqpy_ode"
)
sys.modules[__name__] = ode # mutate myself
