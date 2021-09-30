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
    name_sql ='all_init'

    with global_vals.engine_ali.connect() as conn:
        query = f"SELECT * FROM {table_name} WHERE name_sql like '{name_sql}:%%'"
        df = pd.read_sql(query, conn)
    global_vals.engine_ali.dispose()

    df['factors'] = df['factors'].str.split(', ')
    df['n_factors'] = df['factors'].str.len()
    for name, g in df.groupby(['name_sql']):
        f = [e for x in g['factors'].values for e in x]
        x = dict(Counter(f))
        x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
        plt.barh(y=list(x.keys()), width=list(x.values()))
        plt.title('{}:{}'.format(table_name, name))
        plt.tight_layout()
        plt.show()
        print(df)

if __name__=="__main__":
    # count_plot_from_csv()
    count_plot_from_db()