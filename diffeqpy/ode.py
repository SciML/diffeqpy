import os
import sys

from juliacall import Main

script_dir = os.path.dirname(os.path.realpath(__file__))
Main.include(os.path.join(script_dir, "setup.jl"))

# TODO: find a better way to do this or upstream this function into juliacall
def load_julia_package(name):
    return Main.seval(f"using {name}; {name}")

OrdinaryDiffEq = load_julia_package("OrdinaryDiffEq")
sys.modules[__name__] = OrdinaryDiffEq   # mutate myself
