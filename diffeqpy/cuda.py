import sys
from . import load_julia_packages
cuda, _ = load_julia_packages("DiffEqGPU", "CUDA")
from juliacall import Main
cuda.CUDABackend = Main.seval("CUDA.CUDABackend") # kinda hacky
sys.modules[__name__] = cuda # mutate myself
