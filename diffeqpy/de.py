import os
import sys

from jill.install import install_julia
install_julia(confirm=True)

from julia import Main

script_dir = os.path.dirname(os.path.realpath(__file__))
Main.include(os.path.join(script_dir, "setup.jl"))

from julia import DifferentialEquations
sys.modules[__name__] = DifferentialEquations   # mutate myself
