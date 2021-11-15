import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import global_vals


def count_plot_from_csv():
    for i in [91, 30, 7]:
        # csv_name = f'new2_stepwise_{i}.csv'
        # csv_name = f'hierarchy_average_vol_{i}_cophenetic.csv'
        csv_name = f'hierarchy_cluster_{i}_cophenetic.csv'
        df = pd.read_csv(csv_name)

        df['factors'] = df['factors'].str.split(', ')
        df['n_factors'] = df['factors'].str.len()
        df['dummy'] = True

        for name, g in df.groupby(['dummy']):
            f = [e for x in g['factors'].values for e in x]
            x = dict(Counter(f))
            x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

            plt.barh(y=list(x.keys()), width=list(x.values()))
            plt.title(csv_name)
            plt.tight_layout()
            plt.show()
            print(df)

def count_plot_from_db():
    table_name = 'des_factor_hierarchical'
    # table_name = 'des_factor_fcm'
    # table_name = 'des_factor_gaussian'

    name_sql ='all_init_new'
    # name_sql ='combination'
    # name_sql = 'all_init_max' # for hierarchical
    # name_sql ='all_comb_all'

    with global_vals.engine_ali.connect() as conn:
        query = f"SELECT * FROM {table_name} WHERE name_sql like '{name_sql}:%%'"
        df = pd.read_sql(query, conn)
    global_vals.engine_ali.dispose()

    df['factors'] = df['factors'].str.split(', ')
    df['n_factors'] = df['factors'].str.len()

    # for name, g in df.groupby(['n_factors']):
    #     print(name, g['cophenetic'].mean())

    all_f_r = []
    all_f_10 = []
    for name, g in df.groupby(['name_sql']):
        print(name)
        f = [e for x in g['factors'].values for e in x]
        x = dict(Counter(f))
        x = {k: v/len(g) for k, v in sorted(x.items(), key=lambda item: item[1])}
        all_f_r.extend([k for k, v in x.items() if v>0.2])
        all_f_10.extend(list(x.keys())[-10:])
        plt.barh(y=list(x.keys()), width=list(x.values()))
        plt.axvline(x=0.2, color='r')
        plt.title('{}:{}'.format(table_name, name))
        plt.tight_layout()
        plt.savefig('{}_{}.png'.format(table_name, name))
        plt.close()

    all_f_r = list(set(all_f_r))
    all_f_10 = list(set(all_f_10))

    print('20%:', len(all_f_r), all_f_r)
    print('top10:', len(all_f_10), all_f_10)

def count_plot_from_db_comb_hierarchical():
    table_name = 'des_factor_hierarchical'
    name_sql ='all_comb_new_multiperiod1'
    groupby_col = ['name_sql', 'factors']
    sort_col = 'cophenetic'

    with global_vals.engine_ali.connect() as conn:
        query = f"SELECT * FROM {table_name} WHERE name_sql like '{name_sql}:%%'"
        df = pd.read_sql(query, conn)
    global_vals.engine_ali.dispose()

    sort_df = df.groupby(groupby_col).mean()[sort_col].sort_values(ascending=False).reset_index()
    sort_df['factors'] = sort_df['factors'].str.split(', ')
    sort_df['n_factors'] = sort_df['factors'].str.len()

    best = sort_df.groupby(['name_sql','n_factors']).first()
    print(best)

def count_plot_from_db_comb_fcm():
    table_name = 'des_factor_fcm'
    name_sql ='all_comb_new_multiperiod1'
    groupby_col = ['name_sql', 'factors','n_clusters','m']
    sort_col = 'xie_beni_index'

    with global_vals.engine_ali.connect() as conn:
        query = f"SELECT * FROM {table_name} WHERE name_sql like '{name_sql}:%%'"
        df = pd.read_sql(query, conn)
    global_vals.engine_ali.dispose()

    sort_df = df.groupby(groupby_col).mean()[sort_col].sort_values(ascending=False).reset_index()
    sort_df['factors'] = sort_df['factors'].str.split(', ')
    sort_df['n_factors'] = sort_df['factors'].str.len()

    best = sort_df.groupby(['name_sql','n_factors','n_clusters']).first()
    print(best)

def count_plot_from_db_comb_gaussian():
    table_name = 'des_factor_gaussian'
    name_sql ='all_comb_new_multiperiod1'
    groupby_col = ['name_sql', 'factors','n_clusters']
    sort_col = 'S_Dbw'

    with global_vals.engine_ali.connect() as conn:
        query = f"SELECT * FROM {table_name} WHERE name_sql like '{name_sql}:%%'"
        df = pd.read_sql(query, conn)
    global_vals.engine_ali.dispose()

    sort_df = df.groupby(groupby_col).mean()[sort_col].sort_values(ascending=False).reset_index()
    sort_df['factors'] = sort_df['factors'].str.split(', ')
    sort_df['n_factors'] = sort_df['factors'].str.len()

    best = sort_df.groupby(['name_sql','n_factors','n_clusters']).first()
    print(best)


def count_plot_from_db_final(pillar=None, testing_interval=None, currency=None):
    table_name = 'des_factor_final'
    groupby_col = ['n_factor','cols','testing_interval']
    sort_col = 'score'

    conditions=['True']
    if pillar:
        conditions.append(f"pillar='{pillar}'")
    if testing_interval:
        conditions.append(f"testing_interval={testing_interval}")
    if currency:
        conditions.append(f"currency='{currency}'")

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions)}", conn)
    global_vals.engine_ali.dispose()

    sort_df = df.groupby(groupby_col).mean()[sort_col].sort_values(ascending=False).reset_index()
    sort_df = sort_df.sort_values(by=["score"])
    print(pillar, testing_interval, currency, sort_df[["cols","score"]].head(10))

if __name__=="__main__":
    # count_plot_from_csv()
    # count_plot_from_db()
    # count_plot_from_db_comb_hierarchical()
    # count_plot_from_db_comb_fcm()
    # count_plot_from_db_comb_gaussian()
    count_plot_from_db_final(pillar='funda', currency='USD')