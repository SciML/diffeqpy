import sys
from . import load_julia_packages
amdgpu, _ = load_julia_packages("DiffEqGPU", "AMDGPU")
from juliacall import Main
amdgpu.AMDGPUBackend = Main.seval("AMDGPU.AMDGPUBackend") # kinda hacky
sys.modules[__name__] = amdgpu # mutate myself
