import pandas as pd
import numpy as np
from sqlalchemy import text

import global_vals

r_name = '2'
iter_name = r_name#.split('_')[-1]

def feature_importance():
    ''' donwload results from results_lightgbm '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period FROM {global_vals.feature_importance_table}_rf_class P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_class S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}'")
        df = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    df['name'] = df['name'].replace([x for x in df['name'].to_list() if 'org_' in x], 'org')
    df['name'] = df['name'].replace([x for x in df['name'].to_list() if 'ar_' in x],
                                    [f"{x.split('_')[0]}_{x.split('_')[-1]}" for x in df['name'].to_list() if 'ar_' in x])

    df1 = df.groupby(['name'])['split'].mean()
    df2 = df.groupby(['name','group_code'])['split'].mean().unstack()

    with pd.ExcelWriter(f'feature/rf_importance_{iter_name}.xlsx') as writer:
        df1.sort_values(ascending=False).to_excel(writer, sheet_name='all')
        df2.to_excel(writer, sheet_name='group_code')

if __name__ == "__main__":
    feature_importance()
