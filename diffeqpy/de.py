from . import setup
_jul = setup()

# Suppress many UndefVarError printed out.
# See: https://github.com/JuliaPy/pyjulia/issues/159
#      https://github.com/JuliaDiffEq/diffeqpy/pull/24
try:
    show = _jul.api.show
    _jul.api.show = _jul._call('identity')
    from julia.DifferentialEquations import *
finally:
    _jul.api.show = show
del show

solve = _jul.pysolve
