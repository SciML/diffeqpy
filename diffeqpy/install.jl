using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("OrdinaryDiffEq")
Pkg.add("PythonCall")
Pkg.build("PythonCall")
using DifferentialEquations
using PythonCall
