using DifferentialEquations
const DE = DifferentialEquations

# This is curiously slow
function sde_exercise()
    f = (u,p,t) -> 1.01*u
    g = (u,p,t) -> 0.87*u
    u0 = 0.5
    tspan = (0.0,1.0)
    prob = DE.SDEProblem(f,g,u0,tspan)
    sol = DE.solve(prob,reltol=1e-3,abstol=1e-3)
    return nothing
end

function ode_exercise()
    f = (u,p,t) -> -u
    u0 = 0.5
    tspan = (0., 1.)
    prob = DE.ODEProblem(f, u0, tspan)
    sol = DE.solve(prob)
    return nothing
end

function ode_exercise2()
    f = function(u,p,t)
        x, y, z = u
        sigma, rho, beta = p
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    end
    u0 = [1.0,0.0,0.0]
    tspan = (0., 100.)
    p = [10.0,28.0,8/3]
    prob = DE.ODEProblem(f, u0, tspan, p)
    sol = DE.solve(prob,saveat=0.01)
    return nothing
end

# From ODE docs
function ode_exercise3()
    f(u,p,t) = 1.01*u
    u0 = 1/2
    tspan = (0.0,1.0)
    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    return nothing
end

ode_exercise()
ode_exercise2()
ode_exercise3()
sde_exercise()
