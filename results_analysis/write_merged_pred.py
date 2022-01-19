import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import argparse

from global_vars import *
from sqlalchemy import text
from sqlalchemy.dialects.postgresql.base import DATE, DOUBLE_PRECISION, TEXT, INTEGER, BOOLEAN, TIMESTAMP
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from general.sql_process import (
    read_query,
    upsert_data_to_database,
    uid_maker,
    trucncate_table_in_database,
    delete_data_on_database,
)

stock_pred_dtypes = dict(
    trading_day=DATE,
    factor_name=TEXT,
    group=TEXT,
    factor_weight=INTEGER,
    pred_z=DOUBLE_PRECISION,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)

def download_stock_pred(q, model, name_sql, suffix=None, eval_start_date=None):
    ''' organize cron / last period prediction and write weight to DB '''

    # --------------------------------- Download Predictions ------------------------------------------
    model_record_col = ['y_type', 'neg_factor', 'testing_period', 'cv_number']
    other_group_col = ['tree_type', 'use_pca']

    logging.info('=== Download prediction history ===')
    conditions = [f"S.name_sql like '{name_sql}%'"]
    if eval_start_date:
        conditions.append(f"S.testing_period>='{eval_start_date}'")
    query = text(f'''
            SELECT P.pred, P.actual, P.y_type as factor_name, P.group, {', '.join(['S.'+x for x in other_group_col+model_record_col])} 
            FROM {result_pred_table} P 
            INNER JOIN {result_score_table} S ON ((S.finish_timing=P.finish_timing) AND (S.group_code=P.group)) 
            WHERE {' AND '.join(conditions)}
            ORDER BY S.finish_timing''')
    result_all_all = read_query(query, db_url_read).rename(columns={"testing_period": "trading_day"})
    result_all_all['y_type'] = result_all_all['y_type'].str[1:-1].apply(lambda x: ','.join(sorted(x.split(','))))

    all_current = []
    all_history = []
    for y_type, result_all in result_all_all.groupby('y_type'):
        logging.info(f'=== Generate rank for [{y_type}] with results={result_all.shape}) ===')

        if len(result_all) < 100:   # test run iteration
            continue

        # remove duplicate samples from running twice when testing
        result_all = result_all.drop_duplicates(subset=['group', 'trading_day', 'factor_name', 'cv_number']+other_group_col, keep='last')
        result_all['trading_day'] = pd.to_datetime(result_all['trading_day'])

        result_all_avg = result_all.groupby(['group', 'trading_day'])['actual'].mean()   # actual factor premiums
        dict_neg_factor = result_all[['trading_day', 'group', 'neg_factor']].drop_duplicates(subset=['trading_day', 'group'], keep='last')
        dict_neg_factor = dict_neg_factor.set_index(['group','trading_day'])['neg_factor'].unstack().to_dict()  # negative value columns

        # use average predictions from different validation sets
        result_all = result_all.groupby(['trading_day','factor_name', 'group']+other_group_col)[['pred', 'actual']].mean()

        # --------------------------------- Add Rank & Evaluation Metrics ------------------------------------------

        if isinstance(q, int):    # if define top/bottom q factors as the best/worse
            q = q/len(result_all['factor_name'].unique())
        q_ = [0., q, 1.-q, 1.]

        # rank within current testing_period
        groupby_keys = ['group','trading_day'] + other_group_col
        result_all['factor_weight'] = result_all.groupby(level=groupby_keys)['pred'].transform(lambda x: pd.qcut(x, q=q_, labels=range(len(q_) - 1), duplicates='drop'))

        # calculate pred_z using mean & std of all predictions in entire testing history
        groupby_keys = ['group'] + other_group_col
        result_all = result_all.join(result_all.groupby(level=groupby_keys)['pred'].agg(['mean', 'std']), on=groupby_keys, how='left')
        result_all['pred_z'] = (result_all['pred'] - result_all['mean']) / result_all['std']
        result_all = result_all.drop(['mean', 'std'], axis=1)

        def get_summary_stats_in_group(g):
            ''' Calculate basic evaluation metrics for factors '''

            ret_dict = {}

            max_g = g[g['factor_weight'] == 2]
            min_g = g[g['factor_weight'] == 0]

            ret_dict['max_factor'] = ','.join(list(max_g.index.get_level_values('factor_name').tolist()))
            ret_dict['min_factor'] = ','.join(list(min_g.index.get_level_values('factor_name').tolist()))
            ret_dict['max_ret'] = max_g['actual'].mean()
            ret_dict['min_ret'] = min_g['actual'].mean()
            ret_dict['mae'] = mean_absolute_error(g['pred'], g['actual'])
            ret_dict['mse'] = mean_squared_error(g['pred'], g['actual'])
            ret_dict['r2'] = r2_score(g['pred'], g['actual'])

            return pd.Series(ret_dict)

        result_all_comb = result_all.groupby(level=['group', 'trading_day']+other_group_col).apply(get_summary_stats_in_group)
        result_all_comb = result_all_comb.loc[result_all_comb.index.get_level_values('trading_day')<result_all_comb.index.get_level_values('trading_day').max()]
        result_all_comb[['max_ret','min_ret','mae','mse','r2']] = result_all_comb[['max_ret','min_ret','mae','mse','r2']].astype(float)
        result_all_comb = result_all_comb.join(result_all_avg, on=['group', 'trading_day']).reset_index()
        result_all_comb_mean = result_all_comb.groupby(['group'] + other_group_col).mean().reset_index()
        logging.debug(result_all_comb_mean)
        logging.debug(result_all_comb.groupby(['group'] + other_group_col)['max_ret'].apply(lambda x: x.mean()/x.std()))

        # --------------------------------- Save Local Evaluation ------------------------------------------

        if DEBUG:    # save local for evaluation
            logging.debug(f'=== Save CSV for backtest average ret ===')
            file_name = f'#{model}_pred_{name_sql}_{y_type[:10]}.xlsx'
            writer = pd.ExcelWriter(file_name)
            result_all_comb_mean.to_excel(writer, sheet_name='average', index=False)
            result_all_comb.to_excel(writer, sheet_name='group_time', index=False)
            pd.pivot_table(result_all, index=['group', 'trading_day'], columns=['factor_name'], values=['pred','actual']).to_excel(writer, sheet_name='all')
            writer.save()
            logging.debug(f'=== Saved [{file_name}] for evaluation ===')

            logging.debug(f'=== Save Plot for backtest average ret ===')
            result_all_comb['other_group'] = result_all_comb[other_group_col].astype(str).agg('-'.join, axis=1)
            num_group = len(result_all_comb['group'].unique())
            num_other_group = len(result_all_comb['other_group'].unique())
            fig = plt.figure(figsize=(num_group*8, num_other_group*4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
            k=1
            for name, g in result_all_comb.groupby(['other_group', 'group']):
                ax = fig.add_subplot(num_other_group, num_group, k)
                g[['max_ret','actual','min_ret']] = (g[['max_ret','actual','min_ret']] + 1).cumprod(axis=0)
                plot_df = g[['max_ret','actual','min_ret']]
                ax.plot(plot_df)
                for i in range(3):
                    ax.annotate(plot_df.iloc[-1, i].round(2), (plot_df.index[-1], plot_df.iloc[-1, i]), fontsize=10)
                if k % num_group == 1:
                    ax.set_ylabel(name[0], fontsize=20)
                if k > (num_other_group-1)*num_group:
                    ax.set_xlabel(name[1], fontsize=20)
                if k==1:
                    plt.legend(['best','average','worse'])
                k+=1
            plt.ylabel('-'.join(other_group_col))
            plt.xlabel('group')
            fig_name = f'#{model}_pred_{name_sql}_{y_type[:10]}.png'
            plt.savefig(fig_name)
            plt.close()
            logging.debug(f'=== Saved [{fig_name}] for evaluation ===')

        # ------------------------ Select Best Config (among other_group_col) ------------------------------

        result_all_comb_mean['net_ret'] = result_all_comb_mean['max_ret'] - result_all_comb_mean['min_ret']
        result_all_comb_mean_best = result_all_comb_mean.sort_values(['max_ret']).groupby(['group']).last()[other_group_col].reset_index()
        logging.info(f'best_iter:\n{result_all_comb_mean_best}')

        result_all = result_all.dropna(axis=0, subset=['factor_weight'])[['pred_z','factor_weight']].reset_index()
        result_all = result_all.merge(result_all_comb_mean_best, on=['group']+other_group_col, how='right')

        # --------------------------------- Save Prod Table to DB ------------------------------------------

        # count rank for debugging
        # factor_rank = result_all.set_index(['trading_day','factor_name','group'])['factor_weight'].unstack()
        rank_count = result_all.groupby(['group','trading_day'])['factor_weight'].apply(pd.value_counts)
        rank_count = rank_count.unstack().fillna(0)
        logging.debug(f'n_factor used:\n{rank_count}')

        for period in result_all['trading_day'].unique():
            logging.debug(f'calculate (neg_factor, factor_weight): {period}')
            result_col = ['group','trading_day','factor_name','pred_z','factor_weight']
            df = result_all.loc[result_all['trading_day']==period, result_col].copy().reset_index(drop=True)

            # basic info
            # df['trading_day'] = df['trading_day'] + MonthEnd(1)
            df['factor_weight'] = df['factor_weight'].astype(int)
            df['long_large'] = False             # original premium is "small - big" = short_large -> those marked neg_factor = long_large
            df['last_update'] = dt.datetime.now()

            neg_factor = dict_neg_factor[pd.Timestamp(period)]

            for k, v in neg_factor.items():     # write neg_factor i.e. label factors
                df.loc[(df['group']==k)&(df['factor_name'].isin([x[2:] for x in v.split(',')])), 'long_large'] = True

            # append to history / currency df list
            all_history.append(df.sort_values(['group', 'pred_z']))
            if (period == result_all['trading_day'].max()):  # if keep_all_history also write to prod table
                all_current.append(df.sort_values(['group', 'pred_z']))

    tbl_name_history = production_factor_rank_history_table
    df_history = pd.concat(all_history, axis=0)
    df_history["weeks_to_expire"] = suffix
    df_history = uid_maker(df_history, primary_key=["group","trading_day","factor_name","weeks_to_expire"])
    df_history = df_history.drop_duplicates(subset=["uid"], keep="last")
    if not DEBUG:
        # trucncate_table_in_database(tbl_name_history, db_url_write)
        upsert_data_to_database(df_history, tbl_name_history, primary_key=["uid"], db_url=db_url_write, how='ignore')

    tbl_name_current = production_factor_rank_table
    df_current = pd.concat(all_current, axis=0)
    df_current["weeks_to_expire"] = suffix
    df_current = uid_maker(df_current, primary_key=["group", "factor_name", "weeks_to_expire"])
    df_current = df_current.drop_duplicates(subset=["uid"], keep="last")
    if not DEBUG:
        delete_data_on_database(tbl_name_current, db_url_read, query=f"weeks_to_expire={suffix}")
        upsert_data_to_database(df_current, tbl_name_current, primary_key=["uid"], db_url=db_url_write, how='append')

if __name__ == "__main__":

    # name_sql = 'week4_20220119_debug'
    name_sql = 'week1_20220116'
    suffix = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=float, default=1/3)
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()

    if args.q.is_integer():
        q = int(args.q)
    elif args.q < .5:
        q = args.q
    else:
        raise Exception('q is either >= .5 or not a numeric')

    # Example
    download_stock_pred(
        q,
        args.model,
        name_sql=name_sql,
        suffix=suffix,
        eval_start_date=None,
    )

    # from results_analysis.score_backtest import score_history
    # score_history(suffix)

