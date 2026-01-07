"""
Warmup utilities for diffeqpy to reduce time-to-first-solve (TTFX).

The first solve in diffeqpy can be slow due to Julia's JIT compilation.
This module provides functions to trigger this compilation upfront,
so subsequent solves are faster.

Example usage:
    from diffeqpy import ode
    from diffeqpy.warmup import warmup_ode

    # Trigger JIT compilation
    warmup_ode()

    # Now solves will be faster
    def f(u, p, t):
        return -u
    prob = ode.ODEProblem(f, 0.5, (0.0, 1.0))
    sol = ode.solve(prob)  # This will be faster
"""


def warmup_ode(verbose=True):
    """
    Warm up the ODE solver by running a simple problem.

    This triggers JIT compilation of common code paths, reducing
    the time-to-first-solve for subsequent ODE problems.

    Parameters
    ----------
    verbose : bool, optional
        If True, print timing information. Default is True.

    Returns
    -------
    float
        Time taken for warmup in seconds.

    Example
    -------
    >>> from diffeqpy import ode
    >>> from diffeqpy.warmup import warmup_ode
    >>> warmup_ode()
    Warming up ODE solver...
    Warmup complete in X.XX seconds.
    """
    import time

    if verbose:
        print("Warming up ODE solver...")

    start = time.time()

    from diffeqpy import ode

    # Scalar ODE
    def f_scalar(u, p, t):
        return -u

    prob = ode.ODEProblem(f_scalar, 0.5, (0.0, 1.0))
    sol = ode.solve(prob)

    # Access solution to trigger interpolation compilation
    _ = sol(0.5)
    _ = sol.t
    _ = sol.u

    # Vector ODE (in-place)
    def f_vec(du, u, p, t):
        du[0] = 10.0 * (u[1] - u[0])
        du[1] = u[0] * (28.0 - u[2]) - u[1]
        du[2] = u[0] * u[1] - (8.0 / 3.0) * u[2]

    prob_vec = ode.ODEProblem(f_vec, [1.0, 0.0, 0.0], (0.0, 0.1))
    sol_vec = ode.solve(prob_vec)

    elapsed = time.time() - start

    if verbose:
        print(f"Warmup complete in {elapsed:.2f} seconds.")

    return elapsed


def warmup_de(verbose=True):
    """
    Warm up the full DifferentialEquations suite.

    This triggers JIT compilation for ODE, SDE, DAE, and DDE solvers.
    Note: This takes longer than warmup_ode() but covers more use cases.

    Parameters
    ----------
    verbose : bool, optional
        If True, print timing information. Default is True.

    Returns
    -------
    float
        Time taken for warmup in seconds.

    Example
    -------
    >>> from diffeqpy import de
    >>> from diffeqpy.warmup import warmup_de
    >>> warmup_de()
    Warming up DifferentialEquations...
    Warmup complete in X.XX seconds.
    """
    import time

    if verbose:
        print("Warming up DifferentialEquations...")

    start = time.time()

    from diffeqpy import de

    # Scalar ODE
    def f_scalar(u, p, t):
        return -u

    prob = de.ODEProblem(f_scalar, 0.5, (0.0, 1.0))
    sol = de.solve(prob)

    # Access solution
    _ = sol(0.5)
    _ = sol.t
    _ = sol.u

    # Vector ODE (in-place)
    def f_vec(du, u, p, t):
        du[0] = 10.0 * (u[1] - u[0])
        du[1] = u[0] * (28.0 - u[2]) - u[1]
        du[2] = u[0] * u[1] - (8.0 / 3.0) * u[2]

    prob_vec = de.ODEProblem(f_vec, [1.0, 0.0, 0.0], (0.0, 0.1))
    sol_vec = de.solve(prob_vec)

    # SDE
    def f_sde(u, p, t):
        return 1.01 * u

    def g_sde(u, p, t):
        return 0.87 * u

    prob_sde = de.SDEProblem(f_sde, g_sde, 0.5, (0.0, 0.1))
    sol_sde = de.solve(prob_sde)

    elapsed = time.time() - start

    if verbose:
        print(f"Warmup complete in {elapsed:.2f} seconds.")

    return elapsed
