@debug "Importing DiffEqBase.jl...."
try
    import DiffEqBase
catch err
    @error "Failed to import DiffEqBase.jl" exception = (err, catch_backtrace())
    rethrow()
end

import PythonCall

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
