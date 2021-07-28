from preprocess.load_data import combine_data, download_index_return
import global_vals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_index_return():

    print(f'#################################################################################################')
    print(f'      ------------------------> Download index return data from {global_vals.processed_ratio_table}')

    # db_table_name = global_vals.processed_ratio_table + '_biweekly'
    db_table_name = global_vals.processed_stock_table
    # db_table_name = global_vals.processed_ratio_table

    with global_vals.engine_ali.connect() as conn:
        index_ret = pd.read_sql(f"SELECT * FROM {db_table_name} WHERE ticker like '.%%'", conn)
    global_vals.engine_ali.dispose()

    index_ret = index_ret.loc[index_ret['ticker']!='.HSLI']

    fig = plt.figure(figsize=(12, 12), dpi=120, constrained_layout=True)
    writer = pd.ExcelWriter(f'eda/index_corr.xlsx')

    k=1
    for i in index_ret.columns.to_list()[2:]:
        ax = fig.add_subplot(3, 3, k)
        g = index_ret[['period_end', 'ticker', i]]
        g = g.set_index(['period_end', 'ticker']).unstack()
        g.columns = [x[1] for x in g.columns.to_list()]
        g = g.reset_index().drop(['period_end'], axis=1)

        cr = g.astype(float).corr()
        sns.set(style="whitegrid", font_scale=0.5)
        ax = sns.heatmap(cr, cmap='PiYG', vmin=-1, vmax=1, label='small')
        ax.set_title(i, fontsize=20)
        k += 1

        cr_df = cr.stack(-1)
        cr_df = cr_df.reset_index()
        cr_df.columns=['f1','f2','corr']
        cr_df = cr_df.loc[cr_df['corr']!=1].drop_duplicates(subset=['corr'])
        cr_df['corr_abs'] = cr_df['corr'].abs()
        cr_df = cr_df.sort_values(by=['corr_abs'], ascending=False)
        cr_df.to_excel(writer, sheet_name=i)

    plt.savefig('eda/index_corr.png', dpi=300)
    writer.save()

if __name__ == '__main__':
    df = get_index_return()
    print(df)
