import pandas as pd
import numpy as np
from sqlalchemy import text

import global_vals

r_name = 'lasso_multipca'
pred_table_name = 'lasso'
score_table_name = 'lasso'

iter_name = r_name#.split('_')[-1]
use_pca = False

def feature_importance(use_pca):
    ''' donwload results from results_lightgbm '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.alpha, S.y_type, S.cv_number, S.group_code as group, S.testing_period FROM {global_vals.feature_importance_table}_{pred_table_name} P "
                     f"INNER JOIN {global_vals.result_score_table}_{score_table_name} S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}'")
        df = pd.read_sql(query, conn)       # download training history
        pca = pd.read_sql(f"SELECT * FROM {global_vals.processed_pca_table}", conn)
    global_vals.engine_ali.dispose()

    # df = df.loc[df['alpha']==0.01]

    # df['name'] = df['name'].replace([x for x in df['name'].to_list() if 'org_' in x], 'org')
    # df['name'] = df['name'].replace([x for x in df['name'].to_list() if 'ar_' in x],
    #                                 [f"{x.split('_')[0]}_{x.split('_')[-1]}" for x in df['name'].to_list() if 'ar_' in x])

    df['max'] = df.groupby(['cv_number','group','testing_period','y_type'])['split'].transform(np.nanmax)
    df['split'] = df['split']/df['max']

    if use_pca:
        pca = pca.merge(df[['testing_period','group','split']], on=['testing_period','group'])
        pca.iloc[:, 1:-4] = pca.iloc[:, 1:-4].multiply(pca['split'], axis='index')
        df = pca.groupby(['testing_period','group']).sum()
        df = df.iloc[:-2].stack().reset_index()
        df.columns = ['testing_period','group','name','split']

    # df1 = df.groupby(['name'])['split'].mean()
    df2 = df.groupby(['name','group','alpha'])['split'].mean().unstack()

    with pd.ExcelWriter(f'feature/{score_table_name}_importance_{iter_name}.xlsx') as writer:
        # df1.sort_values(ascending=False).to_excel(writer, sheet_name='all')
        df2.to_excel(writer, sheet_name='group_code')

if __name__ == "__main__":
    feature_importance(use_pca)
