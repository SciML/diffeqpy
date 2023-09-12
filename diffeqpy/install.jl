using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("OrdinaryDiffEq")
Pkg.add("DiffEqBase")
Pkg.develop("PythonCall")
Pkg.build("PythonCall")
using DifferentialEquations
using PythonCall
