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

#=

from diffeqpy import de
import numpy as np

def f(du,u,p,t):
    du[0] = -u[0]

u0 = np.array([0.5])
tspan = (0., 1.)
prob = de.ODEProblem(f, u0, tspan)
sol = de.solve(prob)



import matplotlib.pyplot as plt
plt.plot(sol.t,[u[0] for u in sol.u])
plt.show()



import numpy
t = numpy.linspace(0,1,100)
u = sol(t)
plt.plot(t,[ui[0] for ui in u])
plt.show()


jul_f = de.seval("(u,p,t)->-u") # Define the anonymous function in Julia
prob = de.ODEProblem(jul_f, u0[0], tspan)
sol = de.solve(prob)



def f(du,u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    du[0] = sigma * (y - x)
    du[1] = x * (rho - z) - y
    du[2] = x * y - beta * z

u0 = np.array([1.0,0.0,0.0])
tspan = (0., 100.)
p = (10.0,28.0,8/3)
prob = de.ODEProblem(f, u0, tspan, p)
sol = de.solve(prob,saveat=0.01)
u = np.array([list(ui) for ui in sol.u])

plt.plot(sol.t,u)
plt.show()

ut = numpy.transpose(u)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ut[0,:],ut[1,:],ut[2,:])
plt.show()
=#