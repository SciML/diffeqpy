using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("OrdinaryDiffEq")
Pkg.add("DiffEqBase")
Pkg.add("PyCall")
Pkg.build("PyCall")
using DifferentialEquations
using PyCall
