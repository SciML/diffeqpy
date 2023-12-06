import sys
from . import load_julia_packages
de, _, _ = load_julia_packages("DifferentialEquations", "ModelingToolkit", "PythonCall")
from juliacall import Main
de.jit = Main.seval("jit(x) = typeof(x).name.wrapper(ModelingToolkit.modelingtoolkitize(x), x.u0, x.tspan, x.p)") # kinda hackey
de.jit32 = Main.seval("jit(x) = typeof(x).name.wrapper(ModelingToolkit.modelingtoolkitize(x), Float32.(x.u0), Float32.(x.tspan), Float32.(x.p))") # kinda hackey
sys.modules[__name__] = de # mutate myself
