import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vals
import re
import datetime as dt
from descriptive_factor.report_to_slack import file_to_slack, report_to_slack, report_series_to_slack, report_df_to_slack

suffixes = dt.datetime.today().strftime('%Y%m%d')
SLACK = True
currency_code_list = ["'USD'", "'HKD'"]

class score_eval:
    def __init__(self):
        pass

    def test_history(self):
        ''' test on ai_score history '''
        score_col = ['fundamentals_momentum', 'fundamentals_quality', 'fundamentals_value', 'fundamentals_extra', 'ai_score']

        with global_vals.engine_ali.connect() as conn_ali:
            score_history = pd.read_sql(f"SELECT ticker, period_end, currency_code, stock_return_y, {', '.join(score_col)} "
                                        f"FROM {global_vals.production_score_history} WHERE currency_code in ({','.join(currency_code_list)})", conn_ali)
        global_vals.engine_ali.dispose()

        save_description_history(score_history)
        plot_dist_score(score_history, 'history', score_col)
        qcut_eval(score_col, score_history, name='')

    def test_current(self):
        ''' test on ai_score current '''

        score_col = ['wts_rating', 'dlp_1m', 'fundamentals_momentum', 'fundamentals_quality', 'fundamentals_value',
                     'fundamentals_extra', 'esg', 'ai_score', 'ai_score2']
        pillar_current = {}
        with global_vals.engine_ali.connect() as conn_ali, global_vals.engine.connect() as conn:
            update_time = pd.read_sql(f'SELECT * FROM {global_vals.update_time_table}', conn_ali)
            update_time['update_time'] = update_time['update_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if SLACK:
                report_series_to_slack('*======== Tables Update Time ========*', update_time.set_index('index')['update_time'])
            query = f"SELECT currency_code, S.* FROM {global_vals.production_score_current} S "
            query += f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker "
            query += f"WHERE currency_code in ({','.join(currency_code_list)})"
            score_current = pd.read_sql(query, conn)
            query1 = f"SELECT currency_code, S.* FROM {global_vals.production_score_current_history} S "
            query1 += f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker "
            query1 += f"WHERE currency_code in ({','.join(currency_code_list)})"
            score_current_history = pd.read_sql(query1, conn)
            for p in filter(None, score_current['currency_code'].unique()):
                pillar_current[p] = pd.read_sql(f"SELECT * FROM \"test_fundamental_score_details_{p}\"", conn_ali)
        global_vals.engine_ali.dispose()
        global_vals.engine.dispose()

        # 1. save comparison csv
        score_current_history = score_current_history.loc[score_current_history['trading_day'] < score_current_history['trading_day'].max()]
        score_history_lw = score_current_history.loc[score_current_history['trading_day'] == score_current_history['trading_day'].max()]
        score_history_avg = score_current_history.groupby(['ticker']).mean().reset_index()
        lw_comp = save_compare(score_current, score_history_lw, score_col)
        avg_comp = save_compare(score_current, score_history_avg, score_col)
        lw_comp_des = lw_comp.replace([np.inf, -np.inf],np.nan).describe().transpose()
        avg_comp_des = avg_comp.replace([np.inf, -np.inf],np.nan).describe().transpose()

        writer = pd.ExcelWriter(f'#{suffixes}_compare.xlsx')
        lw_comp_des.to_excel(writer, sheet_name='lastweek_describe (remove inf)')
        lw_comp.to_excel(writer, sheet_name='lastweek')
        avg_comp_des.to_excel(writer, sheet_name='average_describe (remove inf)')
        avg_comp.to_excel(writer, sheet_name='average')
        writer.save()
        
        if SLACK:
            report_series_to_slack('*======== Compare with Last Week (Mean Change) ========*', lw_comp_des['mean'])
            report_series_to_slack('*======== Compare with Score History Average (Mean Change) ========*', avg_comp_des['mean'])
            file_to_slack(f'./#{suffixes}_compare.xlsx', 'xlsx', f'Compare score')

        score_current['ai_score_unscaled'] = score_current[score_col[2:-3]].mean(axis=1)
        score_current['ai_score2_unscaled'] = score_current[score_col[2:-4]+['esg']].mean(axis=1)
        score_col += ['ai_score_unscaled', 'ai_score2_unscaled']

        # 2. test rank
        c1 = score_current.groupby(['currency_code'])['ai_score'].rank(axis=0).corr(score_current.groupby(['currency_code'])['ai_score_unscaled'].rank(axis=0))
        c2 = score_current['ai_score2'].rank(axis=0).corr(score_current['ai_score2_unscaled'].rank(axis=0))
        if SLACK:
            report_to_slack(f'======== ai_score before & after scaler correlation: {round(c1, 3)} ========')
            report_to_slack(f'======== ai_score2 before & after scaler correlation: {round(c2, 3)} ========')

        # 3. save descriptive csv
        save_topn_ticker(score_current)
        save_description(score_current)

        # 4. save descriptive plot
        plot_dist_score(score_current, 'current-DLPA', score_col[:2])
        plot_dist_score(score_current, 'current-fundamentals', score_col[2:-4])
        plot_dist_score(score_current, 'current-final', score_col[-4:])

        plot_minmax_factor(pillar_current)

        self.score_current = score_current

def save_compare(score_cur, score_his, cols):
    df = score_cur.set_index('ticker').merge(score_his.set_index('ticker'), left_index=True, right_index=True, how='outer', suffixes=('_cur','_his'))

    # score changes
    arr = df[[x+'_cur' for x in cols]].values / df[[x+'_his' for x in cols]].values - 1
    r = pd.DataFrame(arr, columns=cols, index=df.index).sort_values(by='ai_score', ascending=True)
    return r

def save_topn_ticker(df, n=20):
    ''' save stock details for top 25 in each score '''

    df = df.loc[df['currency_code'].isin(['HKD','USD'])]

    writer = pd.ExcelWriter(f'#{suffixes}_ai_score_top{n}.xlsx')

    all_df = []
    for i in ['wts_rating', 'dlp_1m', 'ai_score', 'ai_score2']:
        idx = df.groupby(['currency_code'])[i].nlargest(n).index.get_level_values(1)
        ddf = df.loc[idx].sort_values(by=['currency_code',i], ascending=False)
        ddf[['currency_code','ticker',i]].to_excel(writer, sheet_name=i, index=False)
        all_df.append(ddf)

    pd.concat(all_df, axis=0).drop_duplicates().to_excel(writer, sheet_name='original_scores', index=False)
    writer.save()

    if SLACK:
        file_to_slack(f'#{suffixes}_ai_score_top{n}.xlsx', 'xlsx', f'Top {n} tickers')

def save_description(df):
    ''' write statistics for  '''

    df = df.groupby(['currency_code']).agg(['min','mean', 'median', 'max', 'std','count']).transpose()
    print(df)

    writer = pd.ExcelWriter(f'#{suffixes}_describe_current.xlsx')
    df.to_excel(writer, sheet_name='Distribution Current')
    writer.save()

    if SLACK:
        for col in ['ai_score','ai_score_unscaled','ai_score2','ai_score2_unscaled']:
            df_save = df.xs(col, level=0).round(2)
            report_df_to_slack(f'*======== Score Distribution - {col} ========*', df_save)
        file_to_slack(f'#{suffixes}_describe_current.xlsx', 'xlsx', f'{global_vals.production_score_current} Score Distribution')

def save_description_history(df):
    ''' write statistics for description '''

    df = df.groupby(['currency_code','period_end'])['ai_score'].agg(['min','mean', 'median', 'max', 'std','count'])
    print(df)
    writer = pd.ExcelWriter(f'#{suffixes}_describe_history.xlsx')
    df.to_excel(writer, sheet_name='Distribution History')
    writer.save()
    # file_to_slack(f'#{suffixes}_describe_history.xlsx', 'xlsx', f'Backtest Score Distribution')

def plot_dist_score(df, filename, score_col):
    ''' Plot distribution (currency, score)  for all AI score compositions '''

    num_cur = len(df['currency_code'].unique())
    num_score = len(score_col)
    fig = plt.figure(figsize=(num_cur * 4, num_score * 4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
    k=1
    for col in score_col:
        try:
            for name, g in df.groupby(['currency_code']):
                ax = fig.add_subplot(num_score, num_cur, k)
                ax.hist(g[col], bins=20)
                if k % num_cur == 1:
                    ax.set_ylabel(col, fontsize=20)
                if k > (num_score-1)*num_cur:
                    ax.set_xlabel(name, fontsize=20)
                k+=1
        except:
            continue
    plt.suptitle(filename, fontsize=30)
    plt.savefig(f'#{suffixes}_score_dist_{filename}.png')
    # file_to_slack(f'#{suffixes}_score_dist_{filename}.png', 'png', f'{filename} Score Distribution')

def plot_minmax_factor(df_dict):
    ''' plot min/max distribution '''

    for cur, g in df_dict.items():

        cols = g.set_index("index").columns.to_list()
        score_idx = [cols.index(x) for x in cols if re.match("^fundamentals_", x)]+[len(cols)]
        g[cols] = g[cols].apply(pd.to_numeric, errors='coerce')

        n = np.max([j-i for i, j in zip(score_idx[:-1], score_idx[1:])])

        fig = plt.figure(figsize=(n * 4, 20), dpi=120, constrained_layout=True)
        k=1
        row=0
        for col in cols:
            ax = fig.add_subplot(4, n, k)
            try:
                ax.hist(g[col], bins=20)
            except:
                ax.plot(g[col])
            ax.set_xlabel(col, fontsize=20)
            if cols.index(col)+1 in score_idx[1:]:
                row+=1
                k=row*n+1
            else:
                k += 1

        plt.suptitle(cur, fontsize=30)
        fig.savefig(f'#{suffixes}_score_minmax_{cur}.png')
        # file_to_slack(f'#{suffixes}_score_minmax_{cur}.png', 'png', f'{cur} Score Detailed Distribution')
        plt.close(fig)

def qcut_eval(score_col, fundamentals, name=''):
    ''' evaluate score history with 1) descirbe, 2) score 10-qcut mean ret, 3) per period change '''

    writer = pd.ExcelWriter(f'#{suffixes}_score_eval_history_{name}.xlsx')

    best_10 = fundamentals.groupby(['period_end', 'currency_code']).apply(lambda x: x.nlargest(10, columns=['ai_score'], keep='all')['stock_return_y'].mean()).reset_index()
    avg = fundamentals.groupby(['period_end', 'currency_code']).mean().reset_index()
    best_10 = best_10.merge(avg, on=['period_end', 'currency_code']).sort_values(['currency_code','period_end'])
    best_10[[0,'stock_return_y']] = best_10.groupby(['currency_code']).apply(lambda x: (x[[0,'stock_return_y']]+1).cumprod(axis=0))
    best_10.to_excel(writer, sheet_name=f'best10')
    #
    # # 1. Score describe
    # for name, g in fundamentals.groupby(['currency_code']):
    #     df = g.describe().transpose()
    #     df['std'] = df.std()
    #     df.to_excel(writer, sheet_name=f'describe_{name}')

    # 2. Test 10-qcut return
    def score_ret_mean(score_col='ai_score', group_col='currency_code'):
        if group_col=='':
            fundamentals['currency_code'] = 'cross'
            group_col = 'currency_code'
        fundamentals['score_qcut'] = fundamentals.groupby([group_col])[score_col].transform(lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop'))
        mean_ret = pd.pivot_table(fundamentals, index=[group_col], columns=['score_qcut'], values='stock_return_y')
        mean_ret['count'] = fundamentals.groupby([group_col])[score_col].count()
        return mean_ret.transpose()

    for i in score_col:
        df = score_ret_mean(i)
        df.to_excel(writer, sheet_name=f'ret_{i}')
        if i == 'ai_score':
            print(df)
    score_ret_mean('ai_score','').to_excel(writer, sheet_name='ret_cross')

    # 3. Ticker Score Single-Period Change
    fundamentals = fundamentals.sort_values(by=['ticker','period_end'])
    fundamentals['ai_score_last'] = fundamentals.groupby(['ticker'])['ai_score'].shift(1)
    fundamentals['ai_score_change'] = fundamentals['ai_score'] - fundamentals['ai_score_last']
    df = fundamentals.groupby(['ticker'])['ai_score_change'].agg(['mean','std'])
    df.to_excel(writer, sheet_name='score_change')

    writer.save()

    if SLACK:
        file_to_slack(f'#{suffixes}_score_eval_history_{name}.xlsx', 'xlsx', f'Backtest Return')

if __name__ == "__main__":
    eval = score_eval()
    eval.test_current()     # test on universe_rating + test_fundamentals_score_details_{currency}
    eval.test_history()     # test on (history) <-global_vals.production_score_history

    #TODO: descriptive factor (check why 7/30 history worse)