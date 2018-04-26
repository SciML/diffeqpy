import os

from julia import Julia

script_dir = os.path.dirname(os.path.realpath(__file__))


def setup(**kwargs):
    jul = Julia()

    include = jul.eval('include')  # jul.include does not work sometimes
    include(os.path.join(script_dir, 'setup.jl'))
    jl_setup = jul.eval('setup')
    ok = jl_setup(**kwargs)
    if not ok:
        raise RuntimeError('Timeout reached while loading DiffEqPy.jl')

    jul.add_module_functions('DiffEqPy')
    return jul
