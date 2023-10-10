using Pkg
Pkg.activate("diffeqpy", shared=true)
Pkg.add(["DifferentialEquations", "ModelingToolkit", "OrdinaryDiffEq","PythonCall"])
using DifferentialEquations, ModelingToolkit, OrdinaryDiffEq, PythonCall # Precompile
