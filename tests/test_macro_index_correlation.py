from preprocess.load_data import combine_data, download_index_return
import global_vals
import pandas as pd
import numpy as np

def get_index_return():

    print(f'#################################################################################################')
    print(f'      ------------------------> Download index return data from {global_vals.processed_ratio_table}')

    # db_table_name = global_vals.processed_ratio_table + '_biweekly'
    db_table_name = global_vals.processed_stock_table
    # db_table_name = global_vals.processed_ratio_table

    with global_vals.engine_ali.connect() as conn:
        index_ret = pd.read_sql(f"SELECT ticker, return_1yr "
                                f"FROM {db_table_name} WHERE ticker like '.%%'", conn)
    global_vals.engine_ali.dispose()

    stock_return_col = ['stock_return_r1_0', 'stock_return_r6_2', 'stock_return_r12_7']
    index_ret[stock_return_col] = index_ret[stock_return_col] + 1
    index_ret['return_1yr'] = np.prod(index_ret[stock_return_col].values, axis=1) - 1
    index_ret = pd.pivot_table(index_ret, columns=['ticker'], index=['period_end'], values='return').reset_index(drop=False)

if __name__ == '__main__':
    df = get_index_return()
    print(df)
