from . import setup
_jul = setup()
from julia.DifferentialEquations import *
solve = _jul.pysolve
