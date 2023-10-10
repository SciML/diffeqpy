using Pkg
Pkg.activate("diffeqpy", shared=true)
Pkg.add(["DifferentialEquations", "OrdinaryDiffEq", "PythonCall"])
using DifferentialEquations, OrdinaryDiffEq, PythonCall # Precompile
