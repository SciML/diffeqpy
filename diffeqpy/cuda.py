import sys
from . import load_julia_packages
cuda, _ = load_julia_packages("DiffEqGPU, CUDA")
sys.modules[__name__] = cuda # mutate myself
