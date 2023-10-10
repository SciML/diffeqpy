import sys
from . import load_julia_packages
de, _ = load_julia_packages("DifferentialEquations, ModelingToolkit")
from juliacall import Main
de.jit = Main.seval("jit(x) = typeof(x).name.wrapper(ModelingToolkit.modelingtoolkitize(x), x.u0, x.tspan, x.p)") # kinda hackey
sys.modules[__name__] = de # mutate myself
