import sys
from . import load_julia_packages
de, _, _ = load_julia_packages("DifferentialEquations", "ModelingToolkit", "PythonCall")
from juliacall import Main
de.jit = Main.seval("(x) -> typeof(x).name.wrapper(ModelingToolkit.complete(ModelingToolkit.modelingtoolkitize(x); split = false), [], x.tspan)") # kinda hackey
de.jit32 = Main.seval("""
                      (x) -> begin
                          prob = typeof(x).name.wrapper(ModelingToolkit.complete(ModelingToolkit.modelingtoolkitize(x); split = false), [], Float32.(x.tspan))
                          ModelingToolkit.remake(prob; u0 = Float32.(prob.u0), p = Float32.(prob.p))
                      end
                      """) # kinda hackey
sys.modules[__name__] = de # mutate myself
