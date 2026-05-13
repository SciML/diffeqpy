import sys
from . import load_julia_packages
# Match the OrdinaryDiffEq pair declared in diffeqpy/juliapkg.json.
ode, _ = load_julia_packages("OrdinaryDiffEq", "OrdinaryDiffEqDefault")
sys.modules[__name__] = ode # mutate myself
