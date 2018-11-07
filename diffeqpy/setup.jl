import DiffEqBase
import PyCall

PyCall.PyObject(::typeof(DiffEqBase.solve)) =
    PyCall.pyfunctionret(DiffEqBase.solve,Any,Vararg{PyCall.PyAny})

function DiffEqBase.numargs(f::PyCall.PyObject)
    inspect = PyCall.pyimport("inspect")
    haskey(f,:py_func) ? _f = f[:py_func] : _f = f
    if PyCall.pyversion < v"3.0.0"
        return length(first(inspect[:getargspec](_f)))
    else
        return length(first(inspect[:getfullargspec](_f)))
    end
end
