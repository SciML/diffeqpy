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
        joinpath(dirname(dirname(Base.locate_package(
            Base.PkgId(Base.UUID("8f4d0f93-b110-5947-807f-2305c1781a2d"),
                       "Conda")))),
                 "deps", "build.log"),
        String))
end
