import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vals

dlp_col = ['wts_rating', 'dlp_1m']
score_col = ['fundamentals_momentum', 'fundamentals_quality', 'fundamentals_value', 'ai_score']

def read_score_and_eval():
    ''' Read current & historic ai_score and evaluate return & distribution '''

    with global_vals.engine_ali.connect() as conn_ali, global_vals.engine.connect() as conn:
        score_current = pd.read_sql(f"SELECT S.ticker, currency_code, {', '.join(dlp_col + score_col)} FROM {global_vals.production_score_current} S "
                                    f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker WHERE currency_code is not null", conn)
        score_history = pd.read_sql(f"SELECT ticker, period_end, currency_code, stock_return_y, {', '.join(score_col)} FROM {global_vals.production_score_history} WHERE currency_code is not null", conn_ali)
    global_vals.engine_ali.dispose()
    global_vals.engine.dispose()

    plot_dist_dlp_score(score_current, 'current')
    plot_dist_score(score_current, 'current')
    plot_dist_score(score_history, 'history')
    score_eval(score_history)

def plot_dist_score(df, filename):
    ''' Plot distribution (currency, score)  for all AI score compositions '''

    num_cur = len(df['currency_code'].unique())
    num_score = len(score_col)
    fig = plt.figure(figsize=(num_cur * 4, num_score * 4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
    k=1
    for col in score_col:
        for name, g in df.groupby(['currency_code']):
            ax = fig.add_subplot(num_score, num_cur, k)
            ax.hist(g[col], bins=20)
            if k % num_cur == 1:
                ax.set_ylabel(col, fontsize=20)
            if k > (num_score-1)*num_cur:
                ax.set_xlabel(name, fontsize=20)
            k+=1
    plt.suptitle(filename, fontsize=30)
    plt.savefig(f'#score_dist_{filename}.png')

def plot_dist_dlp_score(df, filename):
    ''' Plot distribution (currency, score)  for all DLPA scores '''

    num_cur = len(df['currency_code'].unique())
    num_score = len(dlp_col)
    fig = plt.figure(figsize=(num_cur * 4, num_score * 4), dpi=120, constrained_layout=True)  # create figure for test & train boxplot
    k=1
    for col in dlp_col:
        for name, g in df.groupby(['currency_code']):
            ax = fig.add_subplot(num_score, num_cur, k)
            ax.hist(g[col], bins=10)
            if k % num_cur == 1:
                ax.set_ylabel(col, fontsize=20)
            if k > (num_score-1)*num_cur:
                ax.set_xlabel(name, fontsize=20)
            k+=1
    plt.suptitle(filename + '-DLPA', fontsize=30)
    plt.savefig(f'#score_dist_dlp_{filename}.png')

def score_eval(fundamentals, name=''):
    ''' evaluate score history with 1) descirbe, 2) score 10-qcut mean ret, 3) per period change '''

    writer = pd.ExcelWriter(f'#score_eval_history_{name}.xlsx')

    # 1. Score describe
    for name, g in fundamentals.groupby(['currency_code']):
        df = g.describe().transpose()
        df['std'] = df.std()
        df.to_excel(writer, sheet_name=f'describe_{name}')

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
        score_ret_mean(i).to_excel(writer, sheet_name=f'ret_{i}')
    score_ret_mean('ai_score','').to_excel(writer, sheet_name='ret_cross')

    # 3. Ticker Score Single-Period Change
    fundamentals = fundamentals.sort_values(by=['ticker','period_end'])
    fundamentals['ai_score_last'] = fundamentals.groupby(['ticker'])['ai_score'].shift(1)
    fundamentals['ai_score_change'] = fundamentals['ai_score'] - fundamentals['ai_score_last']
    df = fundamentals.groupby(['ticker'])['ai_score_change'].agg(['mean','std'])
    df.to_excel(writer, sheet_name='score_change')

    writer.save()

if __name__ == "__main__":
    read_score_and_eval()
