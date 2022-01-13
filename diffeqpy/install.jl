using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("DiffEqBase")
Pkg.add("PyCall")
Pkg.build("PyCall")
using DifferentialEquations
using PyCall
