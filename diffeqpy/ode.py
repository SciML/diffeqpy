import os
import sys

from juliacall import Main

# TODO: find a better way to do this or upstream this function into juliacall
def load_julia_package(name):
    return Main.seval(f"using {name}: {name}; {name}")

OrdinaryDiffEq = load_julia_package("OrdinaryDiffEq")
sys.modules[__name__] = OrdinaryDiffEq   # mutate myself
