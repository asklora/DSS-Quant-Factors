import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import datetime as dt
import numpy as np
import ast
import global_vals

r_name = 'biweekly'

y_type = 'market_cap_usd'
iter_name = r_name.split('_')[-1]

def download_stock_pred():
    ''' download training history from score table DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT * FROM {global_vals.result_score_table}_rf_class WHERE name_sql='{r_name}' ORDER BY finish_timing LIMIT 20")
        df = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # df.to_csv(f'score/score_original_{iter_name}.csv', index=False)

    # print('Finish running for factors ', len(set(df['y_type'])), set(df['y_type']))
    # print('Last running for factors ', df['y_type'].to_list()[-1])
    # print(f"Running time {df['finish_timing'].max() - df['finish_timing'].min()}. From {df['finish_timing'].min()} to {df['finish_timing'].max()}")
    y_type_list = df['y_type'][0][1:-1].split(',')
    train_acc_list = [(x, 'train') for x in y_type_list]
    valid_acc_list = [(x, 'valid') for x in y_type_list]
    test_acc_list = [(x, 'test') for x in y_type_list]

    df[train_acc_list] = df['accuracy_train'].apply(lambda x: x[1:-1]).str.split(',', expand=True).astype(float)
    df[valid_acc_list] = df['accuracy_valid'].apply(lambda x: x[1:-1]).str.split(',', expand=True).astype(float)
    df[test_acc_list] = df['accuracy_test'].apply(lambda x: x[1:-1]).str.split(',', expand=True).astype(float)

    writer = pd.ExcelWriter(f'score/rf_result_score_accuracy_{iter_name}.xlsx')

    # 1. select results with best average accuracy_valid_mean
    # best = df.loc[df.groupby(['group_code','testing_period','cv_number'])['accuracy_valid_mean'].transform(max)==df['accuracy_valid_mean']]
    # best = best[['group_code'] + train_acc_list + valid_acc_list + test_acc_list]
    # best_cv_time = best.groupby(['group_code']).mean().transpose()
    #
    # best_id = best_cv_time.index.to_frame().reset_index(drop=True)
    # best_cv_time.index = pd.MultiIndex.from_tuples(best_id[0].tolist())
    # best_cv_time = best_cv_time.transpose().stack(0).reset_index()
    #
    # best_cv_time.to_excel(writer, sheet_name='best_avg_valid_mean', index=False)

    # 2. select results with best accuracy_valid for each y_type
    df_list = []
    for i in range(len(y_type_list)):
        best = df.loc[df.groupby(['group_code','testing_period','cv_number'])[(i, 'valid')].transform(max)==df[(i, 'valid')]]
        best = best[['group_code', (i, 'train'), (i, 'valid'), (i, 'test')]]
        best_cv_time = best.groupby(['group_code']).mean()
        best_cv_time.columns = ['train','valid','test']
        df_list.append(best_cv_time)

    x = pd.concat(df_list,axis=0)
    pd.concat(df_list,axis=0).to_excel(writer, sheet_name='best_each_valid', index=False)
    writer.save()

if __name__ == "__main__":
    download_stock_pred()
