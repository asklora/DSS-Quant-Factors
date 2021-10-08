import pandas as pd
import global_vals
import numpy as np

with global_vals.engine_ali.connect() as conn:
    df = pd.read_sql(
        f"SELECT period_end, \"group\", factor_name, premium FROM processed_factor_premium_monthly_v2 "
        f"WHERE \"group\"='USD' AND NOT trim_outlier ORDER BY period_end", conn, chunksize=10000)
    df = pd.concat(df, axis=0, ignore_index=True)
global_vals.engine_ali.dispose()

def neg_factor_best_period(df, x_col):

    best_best = {}
    for name in x_col:
        best = {}
        g = df[name]
        for i in np.arange(12, 120, 12):
            g['ma'] = g.rolling(i, min_periods=1, closed='left')['premium'].mean()
            g['new_premium'] = np.where(g['ma']>=0, g['premium'], -g['premium'])
            best[i] = g['new_premium'].mean()
        best_best[name] = [k for k, v in best.items() if v==np.max(list(best.values()))][0]

    return best_best

df = neg_factor_best_period(df,['inv_turnover'])

