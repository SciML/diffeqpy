import sys
from . import load_julia_packages
ode, _  = load_julia_packages("OrdinaryDiffEq", "PythonCall")
sys.modules[__name__] = ode # mutate myself
