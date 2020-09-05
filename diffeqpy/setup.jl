import DiffEqBase
import PyCall

PyCall.PyObject(::typeof(DiffEqBase.solve)) =
    PyCall.pyfunctionret(DiffEqBase.solve,Any,Vararg{PyCall.PyAny})

function DiffEqBase.numargs(f::PyCall.PyObject)
    inspect = PyCall.pyimport("inspect")
    PyCall.hasproperty(f,:py_func) ? _f = f.py_func : _f = f
    return length(first(inspect.getfullargspec(_f)))
end
