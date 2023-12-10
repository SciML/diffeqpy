using Pkg
Pkg.activate("diffeqpy", shared=true)
Pkg.add(["DiffEqGPU", "CUDA"])
using DiffEqGPU, CUDA # Precompile
