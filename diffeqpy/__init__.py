import shutil
from jill.install import install_julia

# juliacall must be loaded after `_ensure_julia_installed()` is run,
# so this import is in `load_julia_packages()`
# from juliacall import Main

def _find_julia():
    # TODO: this should probably fallback to query jill
    return shutil.which("julia")

def _ensure_julia_installed():
    if not _find_julia():
        print("No Julia version found. Installing Julia.")
        install_julia(version="1.10")
        if not _find_julia():
            raise RuntimeError(
                "Julia installed with jill but `julia` binary cannot be found in the path"
            )

# TODO: upstream this function or an alternative into juliacall
def load_julia_packages(*names):
    """
    Load Julia packages and return references to them, automatically installing julia and
    the packages as necessary.
    """
    # This is terrifying to many people. However, it seems SciML takes pragmatic approach.
    _ensure_julia_installed()

    script = """import Pkg
    Pkg.activate(\"diffeqpy\", shared=true)
    try
        import {0}
    catch e
        e isa ArgumentError || throw(e)
        Pkg.add([{1}])
        import {0}
    end
    {0}""".format(", ".join(names), ", ".join(f'"{name}"' for name in names))

    # Unfortunately, `seval` doesn't support multi-line strings
    # https://github.com/JuliaPy/PythonCall.jl/issues/433
    script = script.replace("\n", ";")

    # Must be loaded after `_ensure_julia_installed()`
    from juliacall import Main
    return Main.seval(script)



# Deprecated (julia and packages now auto-install)
import os
import subprocess
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))

def install(*, confirm=False):
    """
    Install Julia (if required) and Julia packages required for diffeqpy.
    """
    julia = _find_julia()
    if not julia:
        print("No Julia version found. Installing Julia.")
        install_julia(confirm=confirm)
        julia = _find_julia()
        if not julia:
            raise RuntimeError(
                "Julia installed with jill but `julia` binary cannot be found in the path"
            )
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    subprocess.check_call([julia, os.path.join(script_dir, "deprecated/install.jl")], env=env)

def install_cuda():
    julia = _find_julia()
    if not julia:
        raise RuntimeError(
            "Julia must be installed before adding CUDA. Please run `diffeqpy.install()` first"
        )
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    subprocess.check_call([julia, os.path.join(script_dir, "deprecated/install_cuda.jl")], env=env)

def install_amdgpu():
    julia = _find_julia()
    if not julia:
        raise RuntimeError(
            "Julia must be installed before adding AMDGPU. Please run `diffeqpy.install()` first"
        )
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    subprocess.check_call([julia, os.path.join(script_dir, "deprecated/install_amdgpu.jl")], env=env)

def install_metal():
    julia = _find_julia()
    if not julia:
        raise RuntimeError(
            "Julia must be installed before adding Metal. Please run `deprecated/diffeqpy.install()` first"
        )
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    subprocess.check_call([julia, os.path.join(script_dir, "deprecated/install_metal.jl")], env=env)

def install_oneapi():
    julia = _find_julia()
    if not julia:
        raise RuntimeError(
            "Julia must be installed before adding oneAPI. Please run `diffeqpy.install()` first"
        )
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    subprocess.check_call([julia, os.path.join(script_dir, "deprecated/install_oneapi.jl")], env=env)
