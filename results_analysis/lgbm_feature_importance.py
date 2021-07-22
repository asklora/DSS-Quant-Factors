import pandas as pd
import numpy as np
from sqlalchemy import text

from global_vals import engine, result_score_table

def feature_importance(r_name):

    ''' donwload results from results_lightgbm '''

    with engine.connect() as conn:
        query = text("select group_code, feature_importance from (select DISTINCT *, min(mse_valid) over (partition by group_code, "
                     "testing_period, cv_number) as min_thing from {} where name_sql in :name) t "
                     "where mse_valid = min_thing".format(result_score_table))
        query = query.bindparams(name=tuple(r_name))
        df = pd.read_sql(query, con=conn)  # read records from DB
    engine.dispose()

    df_importance = df['feature_importance'].str.split(',',  expand=True)
    df_importance['group_code'] = df['group_code']
    params =  df_importance.iloc[0,:-1].to_list()
    print(params)

    all_rank_dict = {'all': rank_features(df_importance, params)}
    for name, g in df_importance.groupby('group_code'):
        all_rank_dict[name] = rank_features(g, params)

    pd.DataFrame(all_rank_dict).to_csv('test_feature_importance.csv')

def rank_features(df, params):
    ''' convert slice of df_importance to a ranking of feature importance '''

    rank_dict={}
    for f in params:
        count_importance = np.asarray((df.values.ravel() == f).reshape(df.shape).sum(axis=0)[:-1])/len(df)
        new_df = count_importance.T * np.array(range(1, df.shape[1]))
        rank_dict[f] = new_df.sum(axis=0)

    print(rank_dict)

    return rank_dict

if __name__ == "__main__":
    r_name = ['ibes_yoy_2021-07-08 15:41:09.637260']

    pt = feature_importance(r_name=r_name)
