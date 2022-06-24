import numpy as np

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