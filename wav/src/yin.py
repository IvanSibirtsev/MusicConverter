import numpy as np


def cumulative_mean_normalized_difference_function(frame, tau_max):
    cmndf = np.zeros(tau_max + 1)
    cmndf[0] = 1
    diff_mean = 0

    for tau in range(1, tau_max + 1):
        difference_tau = np.sum((frame[0: -tau] - frame[0 + tau:]) ** 2)
        diff_mean = (diff_mean * (tau - 1) + difference_tau) / tau
        cmndf[tau] = difference_tau / (diff_mean + np.finfo(np.float64).eps)

    return cmndf


def parabolic_interpolation(y1, y2, y3):
    """
    Parabolic interpolation of an extremal value given three samples with equal spacing on the x-axis.
    The middle value y2 is assumed to be the extremal sample of the three.
    Parameters
    ----------
    y1: f(x1)
    y2: f(x2)
    y3: f(x3)
    Returns
    -------
    x_interp: Interpolated x-value (relative to x3-x2)
    y_interp: Interpolated y-value, f(x_interp)
    """

    a = np.finfo(np.float64).eps + (y1 + y3 - 2 * y2) / 2
    b = (y3 - y1) / 2
    x_interp = -b / (2 * a)
    y_interp = y2 - (b ** 2) / (4 * a)

    return x_interp, y_interp