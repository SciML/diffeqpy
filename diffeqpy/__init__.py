import os
import subprocess

from julia import Julia

script_dir = os.path.dirname(os.path.realpath(__file__))


def include(jul, path):
    include = jul.eval('include')  # jul.include does not work sometimes
    return include(os.path.join(script_dir, path))


def setup():
    jul = Julia()
    include(jul, 'setup.jl')

    # Make Julia functions and types exported from
    # DifferentialEquations accessible:
    jul.add_module_functions('DifferentialEquations')

    # pysolve has to be treated manually:
    # See: https://github.com/JuliaPy/pyjulia/issues/117#issuecomment-323498621
    jul.pysolve = jul.eval('pysolve')

    return jul


def install():
    """
    Install Julia packages required for diffeqpy.
    """
    subprocess.check_call(['julia', os.path.join(script_dir, 'install.jl')])
