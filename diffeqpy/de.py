from . import setup
_jul = setup()

# Suppress many UndefVarError printed out.  pyjulia calls
# self.api.jl_call2(self.api.show, stderr, exoc) in
# Julia.check_exception before raising JuliaError.  Although pyjulia
# uses try-except to catch JuliaError, it still calls `self.api.show`
# (which is `Base.show` in Julia).  Here we monkey-patch `Julia.api`
# so that `api.show` points to `tuple` which does not print anything.
# See: https://github.com/JuliaPy/pyjulia/issues/159
#      https://github.com/JuliaDiffEq/diffeqpy/pull/24
_show = _jul.api.show
try:
    _jul.api.show = _jul._call('tuple')  # some "no-op" arity 2 function
    from julia.DifferentialEquations import *
finally:
    _jul.api.show = _show
del _show

solve = _jul.pysolve
