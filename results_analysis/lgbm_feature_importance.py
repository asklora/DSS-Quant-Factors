import pandas as pd
import numpy as np
from sqlalchemy import text

import global_vals

r_name = '2021-07-22 17:46:31.704325_testing'
iter_name = r_name.split('_')[-1]

def feature_importance():
    ''' donwload results from results_lightgbm '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.y_type FROM {global_vals.feature_importance_table}_lgbm_class P "
                     f"INNER JOIN {global_vals.result_score_table}_lgbm_class S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}'")
        df = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    df['name'] = df['']

    df1 = df.groupby(['name', 'y_type'])['split'].mean().unstack()
    df1['avg'] = df1.mean(axis=1)
    df2=df.groupby(['y_type','name','group_code'])['split'].mean().unstack()

    with pd.ExcelWriter(f'feature/importance_{iter_name}.xlsx') as writer:
        df1.sort_values(by=['avg'], ascending=False).to_excel(writer, sheet_name='y_type')
        df2.to_excel(writer, sheet_name='group_code')

if __name__ == "__main__":
    feature_importance()
