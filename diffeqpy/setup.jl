@debug "Importing DiffEqBase.jl...."
try
    import DiffEqBase
catch err
    @error "Failed to import DiffEqBase.jl" exception = (err, catch_backtrace())
    rethrow()
end

import PythonCall

# PythonCall does not implicitly convert the return values of Python functions to Julia
# values. If a user writes a function in Python and passes it to a solver, that function
# will return objects of type Py, but the solver will expect Julia objects. To solve this,
# we add explicit conversion any time a Py is passed into the DifferentialEquations.jl
# ecosystem as a function.

# TODO: upstream this into a package extension between PythonCall and DiffEqBase
wrap(f::PythonCall.Py) = (args...; kws...) -> pyconvert(Any, f(args...; kws...))
using DiffEqBase
DiffEqBase.DAEFunction(f::Py) = DiffEqBase.DAEFunction(wrap(f))
DiffEqBase.ODEFunction{T, U}(f::Py) where {T, U} = DiffEqBase.ODEFunction{T, U}(wrap(f))
# ...
# TODO: make this cover all entrypoints or use some other mechanism to ensure we get every
# function

# function wrap(f::PythonCall.Py)
#     # preserve the number of function arguments because DifferentialEquations uses that to
#     # determine if the function is mutating
#     inspect = PythonCall.pyimport("inspect")
#     PythonCall.hasproperty(f,:py_func) ? _f = f.py_func : _f = f
#     nargs = length(first(inspect.getfullargspec(_f)))
#     if PythonCall.pyconvert(Bool, inspect.ismethod(_f))
#         # `f` is a bound method (i.e., `self.f`) but `getfullargspec`
#         # includes `self` in the `args` list.  So, we are subtracting
#         # 1 manually here:
#         nargs -= 1
#     end
#     @eval (args::Vararg{Any, $nargs}; kw...) -> pyconvert(Any, $f(args...; kw...))
# end


# PyCall.PyObject(::typeof(DiffEqBase.solve)) =
    # PyCall.pyfunctionret(DiffEqBase.solve,Any,Vararg{PyCall.PyAny})

# TODO: upstream this into a package extension between PythonCall and DiffEqBase
function DiffEqBase.numargs(f::PythonCall.Py)
    inspect = PythonCall.pyimport("inspect")
    PythonCall.hasproperty(f,:py_func) ? _f = f.py_func : _f = f
    nargs = length(first(inspect.getfullargspec(_f)))
    if PythonCall.pyconvert(Bool, inspect.ismethod(_f))
        # `f` is a bound method (i.e., `self.f`) but `getfullargspec`
        # includes `self` in the `args` list.  So, we are subtracting
        # 1 manually here:
        return nargs - 1
    else
        return nargs
    end
end

@debug "Importing DifferentialEquationsjl...."
try
    import DifferentialEquations
catch err
    @error "Failed to import DifferentialEquations.jl" exception = (err, catch_backtrace())
    rethrow()
end
