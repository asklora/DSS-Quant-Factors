import pandas as pd
from general.sql.sql_process import read_query, read_table
import global_vars


class main:

    def __init__(self,
                 # name_sql='w4_d-7_20220324031027_debug',
                 name_sql2='best 1/3 model (cluster, adj_mse, org) for each currency'):
        df = read_query(f"SELECT * FROM {global_vars.factor_config_score_table} WHERE name_sql2='{name_sql2}'")
        df['hpot_uid'] = df['uid'].str[20:-1]
        print(df.columns.to_list())

        df = df.loc[df['currency_code'] == "CNY"]  # debug CNY only

        hp = ['bagging_fraction', 'bagging_freq', 'boosting_type', 'feature_fraction', 'lambda_l1', 'lambda_l2',
              'learning_rate', 'max_bin', 'min_data_in_leaf', 'min_gain_to_split', 'num_leaves']
        g_list = []
        for i in hp:
            g = df.groupby(i)[['logloss_train', 'logloss_valid', 'accuracy_test']].agg(['min', 'mean', 'max', 'std'])
            g['para'] = i
            g = g.reset_index().set_index(['para', i])
            # g.index.name = ('')
            g_list.append(g)
        g_all = pd.concat(g_list, axis=0)

        corr = df.corr()

        score_col = df.filter(regex='_train$|_valid$|_test$').columns.to_list()

        df_best = df.sort_values(by='logloss_valid').groupby(['hpot_uid']).first()
        df_best_agg = df_best.groupby(['currency_code', 'pillar'])[score_col].mean()

        corr_best = df_best.corr()

        # df_best_agg.to_csv('save.csv')
        # pass

        df_importance = read_table(global_vars.factor_config_importance_table)
        df_importance['split'] = df_importance.groupby(['uid'])['split'].apply(lambda x: x/x.max())
        df_best_importance = df_importance.merge(df_best[['uid', 'currency_code']], on='uid')

        df_best_importance = df_best_importance.groupby(['index', 'currency_code'])['split'].mean().unstack()
        df_best_importance.to_csv('save1.csv')


if __name__ == '__main__':
    main()
