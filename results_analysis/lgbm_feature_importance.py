import pandas as pd
import numpy as np
from sqlalchemy import text

import global_vals

r_name = 'biweekly_crossar'
r_name = 'biweekly_rerun'

# r_name = 'test_stable9_re'

iter_name = r_name#.split('_')[-1]

def name_mapping(df):

    for col in df.columns.to_list():
        try:
            df.loc[f'ar_0m', col] = df.loc[col, col]
            df.loc[col, col] = np.nan
            for i in [1, 2]:
                df.loc[f'ar_{i}m', col] = df.loc[f'ar_{col}_{i}m', col]
                df.loc[f'ar_{col}_{i}m', col] = np.nan
        except Exception as e:
            print(e)
            continue

    df = df.reset_index()

    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
    global_vals.engine_ali.dispose()
    factor_list = formula.loc[formula['factors'], 'name'].to_list()

    # df['name'] = df['name'].replace([x for x in df['name'].to_list() if 'org_' in x], 'org')
    df['name'] = df['name'].replace([x for x in df['name'].to_list() if 'ma_' in x],
                                    [f"{x[:2]}_{x[-2:]}" for x in df['name'].to_list() if 'ma_' in x])

    df = df.groupby(['name']).mean().reset_index()

    dic = {}
    dic['1-ar'] = [x for x in df['name'] if (x[:3]=='ma_')] + ['ar_0m','ar_1m','ar_2m']
    dic['2-crossar'] = [x for x in df['name'] if (x[:3]=='ar_')&(x not in dic['1-ar'])]
    dic['3-index'] = [x for x in df['name'] if (x[0]=='.')]

    k = 4
    for i in formula['pillar'].unique():
        dic[f'{k}_{i}'] = [x for x in df['name'] if (x in formula.loc[formula['pillar']==i,'name'].to_list())]

    map_dic = {}
    for k, vs in dic.items():
        for v in vs:
            map_dic[v]=k

    df['type'] = df['name'].map(map_dic)
    df['type'] = df['type'].fillna('5-macro')

    return df

def feature_importance():
    ''' donwload results from results_lightgbm '''

    # try:
    #     df = pd.read_csv('f_importance.csv')
    # except Exception as e:
    # print(e)
    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.y_type FROM {global_vals.feature_importance_table}_lgbm_class P "
                     f"INNER JOIN {global_vals.result_score_table}_lgbm_class S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}'")
        df = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()
    # df.to_csv('f_importance.csv', index=False)

    df1 = df.loc[df['group_code']=='currency'].groupby(['name', 'y_type'])['split'].mean().unstack()
    # df1['avg'] = df1.mean(axis=1)
    df1 = name_mapping(df1)

    df11 = df.loc[df['group_code']=='industry'].groupby(['name', 'y_type'])['split'].mean().unstack()
    # df11['avg'] = df11.mean(axis=1)
    df11 = name_mapping(df11)

    df2=df.groupby(['y_type','name','group_code'])['split'].mean().unstack()

    with pd.ExcelWriter(f'feature/lgbm_importance_{iter_name}.xlsx') as writer:
        df1.sort_values(by=['type','name']).to_excel(writer, sheet_name='cur', index=False)
        df1.sort_values(by=['type','name']).groupby(['type']).mean().to_excel(writer, sheet_name='cur_pivot')
        df11.sort_values(by=['type','name']).to_excel(writer, sheet_name='ind', index=False)
        df11.sort_values(by=['type','name']).groupby(['type']).mean().to_excel(writer, sheet_name='ind_pivot')
        df2.to_excel(writer, sheet_name='group_code')

if __name__ == "__main__":
    feature_importance()
