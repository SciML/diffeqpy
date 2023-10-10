import sys
from . import load_julia_packages
sys.modules[__name__] = load_julia_packages("OrdinaryDiffEq") # mutate myself
