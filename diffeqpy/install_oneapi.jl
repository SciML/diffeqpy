using Pkg
Pkg.activate("diffeqpy", shared=true)
Pkg.add(["DiffEqGPU", "oneAPI"])
using DiffEqGPU, oneAPI # Precompile
