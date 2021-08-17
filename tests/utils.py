import numpy as np

from sympy import Eq, solve, sqrt, symbols


def std2series(s, tol=1e-12):
    """Generate a series given a std with ddof=1.

    Parameters
    ----------
    s : float
    tol : float, optional

    Returns
    -------
    series : numpy.array

    Examples
    --------
    >>> std = 0.01
    >>> series = std2series(std)
    >>> std_obtained = np.std(series, ddof=1)
    >>> np.isclose(std_obtained, std)
    True
    """

    x1, x2, x3, s = symbols('x1 x2 x3 s')
    n = 3
    # define std biased formula
    mu = (x1 + x2 + x3) / n
    s_right = sqrt(sum((x-mu)**2 for x in [x1, x2, x3]) / (n-1))
    eq = Eq(s, s_right)

    # build values
    s_v = 0.01
    x2_v, x3_v = (np.random.rand(2) * s_v ** 2) + 10
    values = {s: s_v, x2: x2_v, x3: x3_v}

    # solve equation
    x1_f, x1_f2 = solve(eq, x1)
    x1_v1 = x1_f.evalf(subs=values)
    # x1_v2 = x1_f2.evalf(subs=values)

    # obtain the series
    out = np.array([float(x1_v1), float(x2_v), float(x3_v)])

    # check
    if abs(s_v - np.std(out, ddof=1)) > tol:
        raise ValueError("The series is not good as expected by ``tol``.")

    return out
