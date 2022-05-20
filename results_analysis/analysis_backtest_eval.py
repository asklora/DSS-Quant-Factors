import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import datetime as dt
import numpy as np
from global_vars import *
from general.sql_process import read_query


def download_model(weeks_to_expire='%%', average_days='%%', name_sql=None):
    ''' evaluation runtime calculated metrics '''

    if name_sql:
        query = f"SELECT * FROM {backtest_eval_table} WHERE name_sql like '{name_sql}%%'"
    else:
        query = f"SELECT * FROM {backtest_eval_table} WHERE name_sql like 'w{weeks_to_expire}_d{average_days}_%%'"

    df = read_query(query, db_url_read).fillna(0)
    df['net_ret'] = df['max_ret'] - df['min_ret']
    all_df = pd.concat([df, pd.DataFrame(df['config'].to_list())], axis=1)

    iter_unique_col = ['name_sql', 'group', 'pillar', 'trading_day']  # keep 'cv_number' in last one for averaging
    diff_config_col = list(df['config'].to_list()[0].keys())

    for name, df in all_df.groupby(['group', 'pillar']):

        df_avg_time = df.groupby(iter_unique_col[:-1] + diff_config_col).mean().reset_index().sort_values(by='max_ret', ascending=False)
        continue

        df_best_avg_time = df_avg_time.sort_values(by=["net_ret"], ascending=False).groupby(iter_unique_col[:-1]).first().reset_index()
        df_best_avg_time_y = df_best_avg_time.groupby(iter_unique_col[:-2])['net_ret'].agg(['mean', 'count']).reset_index()
        df_best_avg_time_y['start_time'] = pd.to_datetime(df_best_avg_time_y['name_sql'].str.split('_', expand=True)[2],
                                                      format='%Y%m%d%H%M%S', errors='coerce')
    # df_best_avg_time_y: rank name_sql given best configuration -> to select name_sql for analysis_backtest

    return


def save_plot_backtest_ret(result_all_comb, other_group_col, pillar):
    """ Save Plot for backtest average ret """

    logger.debug(f'=== Save Plot for backtest average ret ===')
    result_all_comb = result_all_comb.copy()
    result_all_comb['other_group'] = result_all_comb[other_group_col].astype(str).agg('-'.join, axis=1)
    result_all_comb['group'] = result_all_comb[['group', 'group_code']].astype(str).agg('-'.join, axis=1)

    num_group = len(result_all_comb['group'].unique())
    num_other_group = len(result_all_comb['other_group'].unique())

    # create figure for test & train boxplot
    fig = plt.figure(figsize=(num_group * 8, num_other_group * 4), dpi=120, constrained_layout=True)
    k = 1
    for (other_group, group), g in result_all_comb.groupby(['other_group', 'group']):
        ax = fig.add_subplot(num_other_group, num_group, k)
        g[['max_ret', 'actual', 'min_ret']] = (g[['max_ret', 'actual', 'min_ret']] + 1).cumprod(axis=0)
        plot_df = g.set_index(['testing_period'])[['max_ret', 'actual', 'min_ret']]
        ax.plot(plot_df)
        for i in range(3):
            ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
        if (k % num_group == 1) or (num_group == 1):
            ax.set_ylabel(other_group, fontsize=20)
        if k > (num_other_group - 1) * num_group:
            ax.set_xlabel(group, fontsize=20)
        if k == 1:
            plt.legend(['best', 'average', 'worse'])
        k += 1
    fig_name = f'#pred_{rank_pred.name_sql}_{pillar}.png'
    plt.suptitle(' - '.join(other_group_col), fontsize=20)
    plt.savefig(fig_name)
    plt.close()
    logger.debug(f'=== Saved [{fig_name}] for evaluation ===')


if __name__ == "__main__":
    # download_model(weeks_to_expire=26, average_days=28)
    download_model(name_sql='w26_d7_20220215152028_debug')