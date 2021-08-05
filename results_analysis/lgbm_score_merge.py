import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import datetime as dt
import numpy as np

import global_vals

r_name = 'lastweekavg_rerun'
model = 'lgbm'

y_type = 'market_cap_usd'
iter_name = r_name

def download_stock_pred():
    ''' download training history from score table DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT * FROM {global_vals.result_score_table}_{model}_class WHERE name_sql='{r_name}'")
        df = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    df = df.sort_values(['finish_timing'])
    # df.to_csv(f'score/score_original_{iter_name}.csv', index=False)

    print('Finish running for factors ', len(set(df['y_type'])), set(df['y_type']))
    print('Last running for factors ', df['y_type'].to_list()[-1])
    print(f"Running time {df['finish_timing'].max() - df['finish_timing'].min()}. From {df['finish_timing'].min()} to {df['finish_timing'].max()}")

    best = df.loc[df.groupby(['y_type','group_code','testing_period','cv_number'])['accuracy_valid'].transform(max) == df['accuracy_valid']]
    best = best[['y_type','group_code','testing_period','cv_number','accuracy_train','accuracy_valid','accuracy_test', 'auc_test', 'train_len','valid_len','test_len']]

    best_cv = best.groupby(['y_type','group_code','testing_period']).mean().reset_index()
    best_cv_time = best.groupby(['y_type','group_code']).mean().reset_index()

    with pd.ExcelWriter(f'score/{model}_score_auc_{iter_name}.xlsx') as writer:
        # df.to_excel(writer, sheet_name='original', index=False)
        best.to_excel(writer, sheet_name='best', index=False)
        best_cv.to_excel(writer, sheet_name='best_avg(cv)', index=False)
        best_cv_time.to_excel(writer, sheet_name='best_avg(cv,time)', index=False)

    return best

def plot_train_v_valid_test(y_type):

    try:
        best = pd.read_csv(f'score/score_original_{iter_name}.csv')
    except Exception as e:
        with global_vals.engine_ali.connect() as conn:
            query = text(f"SELECT * FROM {global_vals.result_score_table}_lgbm_class WHERE name_sql='{r_name}' AND y_type='{y_type}")
            df = pd.read_sql(query, conn)  # download training history
        global_vals.engine_ali.dispose()
        best = df.loc[df.groupby(['y_type', 'group_code', 'testing_period', 'cv_number'])['accuracy_valid'].transform(max) == df['accuracy_valid']]
        best = best[['y_type', 'group_code', 'testing_period', 'cv_number', 'accuracy_train', 'accuracy_valid', 'accuracy_test', 'auc_test','train_len', 'valid_len', 'test_len']]

    subset_a = best[best['group_code']=='currency'].dropna()
    subset_b = best[best['group_code']=='industry'].dropna()

    plt.scatter(subset_a.accuracy_train, subset_a.accuracy_valid, s=10, c='red', label='cur_valid')
    plt.scatter(subset_a.accuracy_train, subset_a.accuracy_test, s=10, c='salmon', label='cur_test')
    plt.scatter(subset_b.accuracy_train, subset_b.accuracy_valid, s=10, c='blue', label='ind_valid')
    plt.scatter(subset_b.accuracy_train, subset_b.accuracy_test, s=10, c='lightblue', label='ind_test')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'scatter_tvt_{y_type}.png')

    # plt.scatter(subset_a.accuracy_valid, subset_a.accuracy_test, s=10, c='red', label='curr')
    # plt.scatter(subset_b.accuracy_valid, subset_b.accuracy_test, s=10, c='blue', label='ind')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    download_stock_pred()
    # plot_train_v_valid_test('market_cap_usd')