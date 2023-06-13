import scipy.special
from scipy.stats import triang
import numpy as np


def get_triangular_distribution(frequency_resolution: int, max_step_cents: int = 50):
    max_step = int(max_step_cents / frequency_resolution)
    triangular_distribution = triang.pdf(np.arange(-max_step, max_step + 1), 0.5, scale=2 * max_step, loc=-max_step)
    return triangular_distribution


def get_beta_distribution(alpha: float, beta: float, thresholds):
    thresholds_indexes = np.arange(len(thresholds))
    beta_distribution = scipy.special.beta(alpha, beta)
    comb = scipy.special.comb(len(thresholds), thresholds_indexes)
    special_beta = scipy.special.beta(thresholds_indexes + alpha, len(thresholds) - thresholds_indexes + beta)
    return comb * special_beta / beta_distribution
