import os
import sys
import shutil

from jill.install import install_julia
from jill.install import last_julia_version

if shutil.which("julia") == None:
  print("No Julia version found. Installing Julia.")
  install_julia(confirm=True)

from julia import Main

script_dir = os.path.dirname(os.path.realpath(__file__))
Main.include(os.path.join(script_dir, "setup.jl"))

from julia import DifferentialEquations
sys.modules[__name__] = DifferentialEquations   # mutate myself
