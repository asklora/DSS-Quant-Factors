import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vals

dlp_col = ['wts_rating', 'dlp_1m']
score_col_current = ['fundamentals_momentum', 'fundamentals_quality', 'fundamentals_value','fundamentals_extra', 'ai_score', 'ai_score2']
score_col_history = ['fundamentals_momentum', 'fundamentals_quality', 'fundamentals_value','fundamentals_extra', 'ai_score']

def read_score_and_eval():
    ''' Read current & historic ai_score and evaluate return & distribution '''

    global score_col_current, score_col_history

    with global_vals.engine_ali.connect() as conn_ali, global_vals.engine.connect() as conn:
        score_current = pd.read_sql(f"SELECT S.ticker, currency_code, {', '.join(dlp_col + score_col_current)} FROM {global_vals.production_score_current} S "
                                    f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker WHERE currency_code is not null", conn)
        score_history = pd.read_sql(f"SELECT ticker, period_end, currency_code, stock_return_y, {', '.join(score_col_history)} FROM {global_vals.production_score_history} WHERE currency_code is not null", conn_ali)
    global_vals.engine_ali.dispose()
    global_vals.engine.dispose()

    # score_history_current = score_history.loc[score_history['period_end']==score_history['period_end'].max()]
    # score_history_current[score_col_history] = score_history_current.groupby(['currency_code'])[score_col_history].rank()
    # score_current[score_col_current] = score_current.groupby(['currency_code'])[score_col_current].rank()
    # score_history_current = score_history_current.merge(score_current, on=['ticker'], suffixes=('_test','_prod'))
    # x = score_history_current.corr()

    save_top25(score_current)
    score_current['ai_score_unscaled'] = score_current[score_col_current[:-1]].mean(axis=1)
    score_col_current += ['ai_score_unscaled']

    save_description(score_current, 'current')
    save_description_period(score_history, 'history')

    plot_dist_dlp_score(score_current, 'current')
    plot_dist_score(score_current, 'current', score_col_current)
    plot_dist_score(score_history, 'history', score_col_history)
    score_eval(score_history, score_history)

def save_top15(df):
    ''' save stock details for top 25 in each score '''

    df = df.loc[df['currency_code'].isin(['HKD','USD'])]

    writer = pd.ExcelWriter(f'#ai_score_top25.xlsx')

    for i in ['wts_rating', 'dlp_1m', 'ai_score', 'ai_score2']:
        idx = df.groupby(['currency_code'])[i].nlargest(n=25).index.get_level_values(1)
        ddf = df.loc[idx].sort_values(by=['currency_code',i], ascending=False)
        ddf[['currency_code','ticker',i]].to_excel(writer, sheet_name=i, index=False)
        try:
            all_df.append(ddf)
        except:
            all_df = ddf.copy()

    all_df.drop_duplicates().to_excel(writer, sheet_name='original_scores', index=False)
    writer.save()

def save_description(df, filename):
    df = df.groupby(['currency_code']).agg(['min','mean', 'median', 'max', 'std']).transpose()
    print(df)
    df.to_csv(f'describe_{filename}.csv')

def save_description_period(df, filename):
    df.groupby(['currency_code','period_end'])['ai_score'].agg(['min','mean', 'median', 'max', 'std']).to_csv(f'describe_period_{filename}.csv')

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

def score_eval(score_col, fundamentals, name=''):
    ''' evaluate score history with 1) descirbe, 2) score 10-qcut mean ret, 3) per period change '''

    writer = pd.ExcelWriter(f'#score_eval_history_{name}.xlsx')

    best_10 = fundamentals.groupby(['period_end', 'currency_code']).apply(lambda x: x.nlargest(10, columns=['ai_score'], keep='all')['stock_return_y'].mean()).reset_index()
    avg = fundamentals.groupby(['period_end', 'currency_code']).mean().reset_index()
    best_10 = best_10.merge(avg, on=['period_end', 'currency_code']).sort_values(['currency_code','period_end'])
    best_10[[0,'stock_return_y']] = best_10.groupby(['currency_code']).apply(lambda x: (x[[0,'stock_return_y']]+1).cumprod(axis=0))
    best_10.to_excel(writer, sheet_name=f'best10')

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

if __name__ == "__main__":
    read_score_and_eval()
