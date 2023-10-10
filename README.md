# diffeqpy

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CI](https://github.com/SciML/diffeqpy/workflows/CI/badge.svg)](https://github.com/SciML/diffeqpy/actions)

diffeqpy is a package for solving differential equations in Python. It utilizes
[DifferentialEquations.jl](http://diffeq.sciml.ai/dev/) for its core routines
to give high performance solving of many different types of differential equations,
including:

- Discrete equations (function maps, discrete stochastic (Gillespie/Markov)
  simulations)
- Ordinary differential equations (ODEs)
- Split and Partitioned ODEs (Symplectic integrators, IMEX Methods)
- Stochastic ordinary differential equations (SODEs or SDEs)
- Random differential equations (RODEs or RDEs)
- Differential algebraic equations (DAEs)
- Delay differential equations (DDEs)
- Mixed discrete and continuous equations (Hybrid Equations, Jump Diffusions)

directly in Python.

If you have any questions, or just want to chat about solvers/using the package,
please feel free to chat in the [Gitter channel](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge). For bug reports, feature requests, etc., please submit an issue.

## Installation

To install diffeqpy, use pip:

```
pip install diffeqpy
```

Using diffeqpy requires that Julia is installed and in the path, along
with DifferentialEquations.jl and PyCall.jl. To install Julia,
download a generic binary from
[the JuliaLang site](https://julialang.org/downloads/) and add it to your path.
To install Julia packages required for diffeqpy, open up Python
interpreter then run:

```pycon
>>> import diffeqpy
>>> diffeqpy.install()
```

and you're good!

## General Flow

Import and setup the solvers available in *DifferentialEquations.jl* via the command:

```py
from diffeqpy import de
```
In case only the solvers available in *OrdinaryDiffEq.jl* are required then use the command:
```py
from diffeqpy import ode
```
The general flow for using the package is to follow exactly as would be done
in Julia, except add `de.` or `ode.` in front. Note that `ode.` has lesser loading time and a smaller memory footprint compared to `de.`.
Most of the commands will work without any modification. Thus
[the DifferentialEquations.jl documentation](https://github.com/SciML/DifferentialEquations.jl)
and the [DiffEqTutorials](https://github.com/SciML/DiffEqTutorials.jl)
are the main in-depth documentation for this package. Below we will show how to
translate these docs to Python code.

## Note about !

Python does not allow `!` in function names, so this is also [a limitation of pyjulia](https://pyjulia.readthedocs.io/en/latest/limitations.html#mismatch-in-valid-set-of-identifiers)
To use functions which on the Julia side have a `!`, like `step!`, replace `!` by `_b`, for example:

```py
from diffeqpy import de

def f(u,p,t):
    return -u

u0 = 0.5
tspan = (0., 1.)
prob = de.ODEProblem(f, u0, tspan)
integrator = de.init(prob, de.Tsit5())
de.step_b(integrator)
```

is valid Python code for using [the integrator interface](https://diffeq.sciml.ai/stable/basics/integrator/).


## Ordinary Differential Equation (ODE) Examples

### One-dimensional ODEs

```py
from diffeqpy import de

def f(u,p,t):
    return -u

u0 = 0.5
tspan = (0., 1.)
prob = de.ODEProblem(f, u0, tspan)
sol = de.solve(prob)
```

The solution object is the same as the one described
[in the DiffEq tutorials](http://diffeq.sciml.ai/dev/tutorials/ode_example#Step-3:-Analyzing-the-Solution-1)
and in the [solution handling documentation](http://diffeq.sciml.ai/dev/basics/solution)
(note: the array interface is missing). Thus for example the solution time points
are saved in `sol.t` and the solution values are saved in `sol.u`. Additionally,
the interpolation `sol(t)` gives a continuous solution.

We can plot the solution values using matplotlib:

```py
import matplotlib.pyplot as plt
plt.plot(sol.t,sol.u)
plt.show()
```

![f1](https://user-images.githubusercontent.com/1814174/39089385-e898e116-457a-11e8-84c6-fb1f4dd82c48.png)

We can utilize the interpolation to get a finer solution:

```py
import numpy
t = numpy.linspace(0,1,100)
u = sol(t)
plt.plot(t,u)
plt.show()
```

![f2](https://user-images.githubusercontent.com/1814174/39089386-e8affac2-457a-11e8-8c35-d0f039803ff8.png)

### Solve commands

The [common interface arguments](http://diffeq.sciml.ai/dev/basics/common_solver_opts)
can be used to control the solve command. For example, let's use `saveat` to
save the solution at every `t=0.1`, and let's utilize the `Vern9()` 9th order
Runge-Kutta method along with low tolerances `abstol=reltol=1e-10`:

```py
sol = de.solve(prob,de.Vern9(),saveat=0.1,abstol=1e-10,reltol=1e-10)
```

The set of algorithms for ODEs is described
[at the ODE solvers page](http://diffeq.sciml.ai/dev/solvers/ode_solve).

### Compilation with `de.jit` and Julia

When solving a differential equation, it's pertinent that your derivative
function `f` is fast since it occurs in the inner loop of the solver. We can
convert the entire ode problem to symbolic form, optimize that symbolic form,
and emit efficient native code to simulate it using `de.jit` to improve the
efficiency of the solver at the expense of added setup time:

```py
fast_prob = de.jit(prob)
sol = de.solve(fast_prob)
```

Additionally, you can directly define the functions in Julia. This will also
allow for specialization and could be helpful to increase the efficiency for
repeat or long calls. This is done via `seval`:

```py
jul_f = de.seval("(u,p,t)->-u") # Define the anonymous function in Julia
prob = de.ODEProblem(jul_f, u0, tspan)
sol = de.solve(prob)
```

#### Note that when using `de.jit`, certain undocumented restrictions apply!!

### Systems of ODEs: Lorenz Equations

To solve systems of ODEs, simply use an array as your initial condition and
define `f` as an array function:

```py
def f(u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

u0 = [1.0,0.0,0.0]
tspan = (0., 100.)
p = [10.0,28.0,8/3]
prob = de.ODEProblem(f, u0, tspan, p)
sol = de.solve(prob,saveat=0.01)

plt.plot(sol.t,de.transpose(de.stack(sol.u)))
plt.show()
```

![f3](https://user-images.githubusercontent.com/1814174/39089387-e8c5d9d2-457a-11e8-8f77-eecfc955ce27.png)

or we can draw the phase plot:

```py
us = de.stack(sol.u)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(us[0,:],us[1,:],us[2,:])
plt.show()
```

![f4](https://user-images.githubusercontent.com/1814174/39089388-e8dae00c-457a-11e8-879f-8b01e0b47178.png)

### In-Place Mutating Form

When dealing with systems of equations, in many cases it's helpful to reduce
memory allocations by using mutating functions. In diffeqpy, the mutating
form adds the mutating vector to the front. Let's make a fast version of the
Lorenz derivative, i.e. mutating and JIT compiled:

```py
def f(du,u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    du[0] = sigma * (y - x)
    du[1] = x * (rho - z) - y
    du[2] = x * y - beta * z

u0 = [1.0,0.0,0.0]
tspan = (0., 100.)
p = [10.0,28.0,2.66]
prob = de.ODEProblem(f, u0, tspan, p)
jit_prob = de.jit(prob)
sol = de.solve(jit_prob)
```

or using a Julia function:

```py
jul_f = de.seval("""
function f(du,u,p,t)
  x, y, z = u
  sigma, rho, beta = p
  du[1] = sigma * (y - x)
  du[2] = x * (rho - z) - y
  du[3] = x * y - beta * z
end""")
u0 = [1.0,0.0,0.0]
tspan = (0., 100.)
p = [10.0,28.0,2.66]
prob = de.ODEProblem(jul_f, u0, tspan, p)
sol = de.solve(prob)
```

## Stochastic Differential Equation (SDE) Examples

### One-dimensional SDEs

Solving one-dimensonal SDEs `du = f(u,t)dt + g(u,t)dW_t` is like an ODE except
with an extra function for the diffusion (randomness or noise) term. The steps
follow the [SDE tutorial](http://diffeq.sciml.ai/dev/tutorials/sde_example).

```py
def f(u,p,t):
  return 1.01*u

def g(u,p,t):
  return 0.87*u

u0 = 0.5
tspan = (0.0,1.0)
prob = de.SDEProblem(f,g,u0,tspan)
sol = de.solve(prob,reltol=1e-3,abstol=1e-3)

plt.plot(sol.t,de.stack(sol.u))
plt.show()
```

![f5](https://user-images.githubusercontent.com/1814174/39089389-e8f0343e-457a-11e8-87bb-9ed152caee02.png)

### Systems of SDEs with Diagonal Noise

An SDE with diagonal noise is where a different Wiener process is applied to
every part of the system. This is common for models with phenomenological noise.
Let's add multiplicative noise to the Lorenz equation:

```py
def f(du,u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    du[0] = sigma * (y - x)
    du[1] = x * (rho - z) - y
    du[2] = x * y - beta * z

def g(du,u,p,t):
    du[0] = 0.3*u[0]
    du[1] = 0.3*u[1]
    du[2] = 0.3*u[2]

u0 = [1.0,0.0,0.0]
tspan = (0., 100.)
p = [10.0,28.0,2.66]
prob = de.jit(de.SDEProblem(f, g, u0, tspan, p))
sol = de.solve(prob)

# Now let's draw a phase plot

us = de.stack(sol.u)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(us[0,:],us[1,:],us[2,:])
plt.show()
```

![f6](https://user-images.githubusercontent.com/1814174/39089390-e906c1ea-457a-11e8-8fd2-5cf059e2165a.png)

### Systems of SDEs with Non-Diagonal Noise

In many cases you may want to share noise terms across the system. This is
known as non-diagonal noise. The
[DifferentialEquations.jl SDE Tutorial](http://diffeq.sciml.ai/dev/tutorials/sde_example#Example-4:-Systems-of-SDEs-with-Non-Diagonal-Noise-1)
explains how the matrix form of the diffusion term corresponds to the
summation style of multiple Wiener processes. Essentially, the row corresponds
to which system the term is applied to, and the column is which noise term.
So `du[i,j]` is the amount of noise due to the `j`th Wiener process that's
applied to `u[i]`. We solve the Lorenz system with correlated noise as follows:

```py
def f(du,u,p,t):
  x, y, z = u
  sigma, rho, beta = p
  du[0] = sigma * (y - x)
  du[1] = x * (rho - z) - y
  du[2] = x * y - beta * z

def g(du,u,p,t):
  du[0,0] = 0.3*u[0]
  du[1,0] = 0.6*u[0]
  du[2,0] = 0.2*u[0]
  du[0,1] = 1.2*u[1]
  du[1,1] = 0.2*u[1]
  du[2,1] = 0.3*u[1]


u0 = [1.0,0.0,0.0]
tspan = (0.0,100.0)
p = [10.0,28.0,2.66]
nrp = numpy.zeros((3,2))
prob = de.SDEProblem(f,g,u0,tspan,p,noise_rate_prototype=nrp)
jit_prob = de.jit(prob)
sol = de.solve(jit_prob,saveat=0.005)

# Now let's draw a phase plot

us = de.stack(sol.u)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(us[0,:],us[1,:],us[2,:])
plt.show()
```

![f7](https://user-images.githubusercontent.com/1814174/39089391-e91f0494-457a-11e8-860a-865caa26c262.png)

Here you can see that the warping effect of the noise correlations is quite visible!

## Differential-Algebraic Equation (DAE) Examples

A differential-algebraic equation is defined by an implicit function
`f(du,u,p,t)=0`. All of the controls are the same as the other examples, except
here you define a function which returns the residuals for each part of the
equation to define the DAE. The initial value `u0` and the initial derivative
`du0` are required, though they do not necessarily have to satisfy `f` (known
as inconsistent initial conditions). The methods will automatically find
consistent initial conditions. In order for this to occur, `differential_vars`
must be set. This vector states which of the variables are differential (have a
derivative term), with `false` meaning that the variable is purely algebraic.

This example shows how to solve the Robertson equation:

```py
def f(du,u,p,t):
  resid1 = - 0.04*u[0]               + 1e4*u[1]*u[2] - du[0]
  resid2 = + 0.04*u[0] - 3e7*u[1]**2 - 1e4*u[1]*u[2] - du[1]
  resid3 = u[0] + u[1] + u[2] - 1.0
  return [resid1,resid2,resid3]

u0 = [1.0, 0.0, 0.0]
du0 = [-0.04, 0.04, 0.0]
tspan = (0.0,100000.0)
differential_vars = [True,True,False]
prob = de.DAEProblem(f,du0,u0,tspan,differential_vars=differential_vars)
sol = de.solve(prob)
```

![f8](https://user-images.githubusercontent.com/1814174/39089392-e932f012-457a-11e8-9979-c006bcfabf71.png)

and the in-place JIT compiled form:

```py
def f(resid,du,u,p,t):
  resid[0] = - 0.04*u[0]               + 1e4*u[1]*u[2] - du[0]
  resid[1] = + 0.04*u[0] - 3e7*u[1]**2 - 1e4*u[1]*u[2] - du[1]
  resid[2] = u[0] + u[1] + u[2] - 1.0

prob = de.DAEProblem(f,du0,u0,tspan,differential_vars=differential_vars)
jit_prob = de.jit(prob) # Error: no method matching matching modelingtoolkitize(::SciMLBase.DAEProblem{...})
sol = de.solve(jit_prob)
```

## Delay Differential Equations

A delay differential equation is an ODE which allows the use of previous values.
In this case, the function needs to be a JIT compiled Julia function. It looks
just like the ODE, except in this case there is a function `h(p,t)` which allows
you to interpolate and grab previous values.

We must provide a history function `h(p,t)` that gives values for `u` before `t0`.
Here we assume that the solution was constant before the initial time point.
Additionally, we pass `constant_lags = [20.0]` to tell the solver that only
constant-time lags were used and what the lag length was. This helps improve
the solver accuracy by accurately stepping at the points of discontinuity.
Together this is:

```py
f = de.seval("""
function f(du, u, h, p, t)
  du[1] = 1.1/(1 + sqrt(10)*(h(p, t-20)[1])^(5/4)) - 10*u[1]/(1 + 40*u[2])
  du[2] = 100*u[1]/(1 + 40*u[2]) - 2.43*u[2]
end""")
u0 = [1.05767027/3, 1.030713491/3]

h = de.seval("""
function h(p,t)
  [1.05767027/3, 1.030713491/3]
end
""")

tspan = (0.0, 100.0)
constant_lags = [20.0]
prob = de.DDEProblem(f,u0,h,tspan,constant_lags=constant_lags)
sol = de.solve(prob,saveat=0.1)

u1 = [sol.u[i][0] for i in range(0,len(sol.u))]
u2 = [sol.u[i][1] for i in range(0,len(sol.u))]

import matplotlib.pyplot as plt
plt.plot(sol.t,u1)
plt.plot(sol.t,u2)
plt.show()
```

![dde](https://user-images.githubusercontent.com/1814174/39229670-815f2eba-4818-11e8-9ba3-a4f61cc845c5.png)

Notice that the solver accurately is able to simulate the kink (discontinuity)
at `t=20` due to the discontinuity of the derivative at the initial time point!
This is why declaring discontinuities can enhance the solver accuracy.

## Known Limitations

- Autodiff does not work on Python functions. When applicable, either define the derivative function
  as a Julia function or set the algorithm to use finite differencing, i.e. `Rodas5(autodiff=false)`.
  All default methods use autodiff.
- Delay differential equations have to use Julia-defined functions otherwise the history function is
  not appropriately typed with the overloads.

## Testing

Unit tests can be run by [`tox`](http://tox.readthedocs.io).

```sh
tox
```

### Troubleshooting

In case you encounter silent failure from `tox`, try running it with
`-- -s` (e.g., `tox -e py36 -- -s`) where `-s` option (`--capture=no`,
i.e., don't capture stdio) is passed to `py.test`.  It may show an
error message `"error initializing LibGit2 module"`.  In this case,
setting environment variable `SSL_CERT_FILE` may help; e.g., try:

```sh
SSL_CERT_FILE=PATH/TO/cert.pem tox -e py36
```

See also: [julia#18693](https://github.com/JuliaLang/julia/issues/18693).
