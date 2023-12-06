import sys
from . import load_julia_packages
oneapi, _ = load_julia_packages("DiffEqGPU", "oneAPI")
from juliacall import Main
oneapi.oneAPIBackend = Main.seval("oneAPI.oneAPIBackend") # kinda hacky
sys.modules[__name__] = oneapi # mutate myself
