import numpy as np
from utils import sys_logger
from .configs import LOGGER_LEVELS
logger = sys_logger(__name__, LOGGER_LEVELS.ANALYSIS_BEST_LASSO_PERIOD)

def find_lasso_best_period(df, x_col):
    best_best = {}
    for name in x_col:
        best = {}
        g = df[name]
        for i in np.arange(12, 120, 12):
            g['ma'] = g.rolling(i, min_periods = 1, closed = 'left')['premium'].mean()
            g['new_premium'] = np.where(g['ma'] >= 0, g['premium'], -g['premium'])
            best[i] = g['new_premium'].mean()

        best_best[name] = [k for k, v in best.items() if v == np.max(list(best.values()))][0]

    return best_best