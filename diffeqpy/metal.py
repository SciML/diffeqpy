import sys
from . import load_julia_packages
metal, _ = load_julia_packages("DiffEqGPU", "Metal")
from juliacall import Main
metal.MetalBackend = Main.seval("Metal.MetalBackend") # kinda hacky
sys.modules[__name__] = metal # mutate myself