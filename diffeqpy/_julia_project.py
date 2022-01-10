import julia
import logging

from julia_project import JuliaProject

import os
diffeqpy_path = os.path.dirname(os.path.abspath(__file__))

julia_project = JuliaProject(
    name="diffeqpy",
    package_path=diffeqpy_path,
    preferred_julia_versions = ['1.7', '1.6', 'latest'],
    env_prefix = 'DIFFEQPY_',
    logging_level = logging.INFO, # or logging.WARN,
    console_logging=False
)

julia_project.run()

# logger = julia_project.logger

def compile_diffeqpy():
    """
    Compile a system image for `diffeqpy` in the subdirectory `./sys_image/`. This
    system image will be loaded the next time you import `diffeqpy`.
    """
    julia_project.compile_julia_project()


def update_diffeqpy():
    """
    Remove possible stale Manifest.toml files and compiled system image.
    Update Julia packages and rebuild Manifest.toml file.
    Before compiling, it's probably a good idea to call this method, then restart Python.
    """
    julia_project.update()


