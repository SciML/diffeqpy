using Pkg
Pkg.activate("diffeqpy", shared=true)
Pkg.add(["DiffEqGPU", "AMDGPU"])
using DiffEqGPU, AMDGPU # Precompile
