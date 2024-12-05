import sys
from . import load_julia_packages
de, _, _ = load_julia_packages("DifferentialEquations", "ModelingToolkit", "PythonCall")
from juliacall import Main
de.seval("jit(x) = typeof(x).name.wrapper(Main.ModelingToolkit.complete(Main.ModelingToolkit.modelingtoolkitize(x); split = false), [], x.tspan)") # kinda hackey
de.seval("""
                      function jit(x)
                          prob = typeof(x).name.wrapper(Main.ModelingToolkit.complete(Main.ModelingToolkit.modelingtoolkitize(x); split = false), [], Float32.(x.tspan))
                          Main.ModelingToolkit.remake(prob; u0 = Float32.(prob.u0), p = Float32.(prob.p))
                      end
                      """) # kinda hackey
sys.modules[__name__] = de # mutate myself
