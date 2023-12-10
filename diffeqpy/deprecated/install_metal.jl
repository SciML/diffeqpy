using Pkg
Pkg.activate("diffeqpy", shared=true)
Pkg.add(["DiffEqGPU", "Metal"])
using DiffEqGPU, Metal # Precompile
