using Pkg
Pkg.add([
    "DifferentialEquations",
    "DiffEqBase",
    "PyCall",
])
using DifferentialEquations
using PyCall

if lowercase(get(ENV, "CI", "false")) == "true"
    @info "PyCall/deps/build.log:"
    print(read(
        joinpath(dirname(dirname(pathof(PyCall))), "deps", "build.log"),
        String))
    @info "Conda/deps/build.log:"
    print(read(
        joinpath(dirname(dirname(Base.find_package("Conda"))),
                 "deps", "build.log"),
        String))
end
