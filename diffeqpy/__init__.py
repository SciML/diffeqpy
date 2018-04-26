import os

from julia import Julia

script_dir = os.path.dirname(os.path.realpath(__file__))


def setup():
    jul = Julia()

    # Load ./setup.jl
    include = jul.eval('include')  # jul.include does not work sometimes
    include(os.path.join(script_dir, 'setup.jl'))

    # Make Julia functions and types exported from
    # DifferentialEquations accessible:
    jul.add_module_functions('DifferentialEquations')

    # pysolve has to be treated manually:
    # See: https://github.com/JuliaPy/pyjulia/issues/117#issuecomment-323498621
    jul.pysolve = jul.eval('pysolve')

    return jul
