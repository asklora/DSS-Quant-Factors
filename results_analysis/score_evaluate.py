import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vals
import re
import datetime as dt

suffixes = dt.datetime.today().strftime('%Y%m%d')

class score_eval:
    def __init__(self):
        pass

    def test_history(self):
        ''' test on ai_score history '''
        score_col = ['fundamentals_momentum', 'fundamentals_quality', 'fundamentals_value', 'fundamentals_extra', 'ai_score']

        with global_vals.engine_ali.connect() as conn_ali:
            score_history = pd.read_sql(f"SELECT ticker, period_end, currency_code, stock_return_y, {', '.join(score_col)} "
                                        f"FROM {global_vals.production_score_history} WHERE currency_code is not null", conn_ali)
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
            query = f"SELECT currency_code, S.* FROM {global_vals.production_score_current} S "
            query += f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker "
            query += f"WHERE currency_code is not null"
            score_current = pd.read_sql(query, conn)
            query1 = f"SELECT currency_code, S.* FROM {global_vals.production_score_current_history} S "
            query1 += f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker "
            query1 += f"WHERE currency_code is not null"
            score_current_history = pd.read_sql(query1, conn)
            for p in filter(None, score_current['currency_code'].unique()):
                pillar_current[p] = pd.read_sql(f"SELECT * FROM \"test_fundamental_score_details_{p}\"", conn_ali)
        global_vals.engine_ali.dispose()
        global_vals.engine.dispose()

        score_current['ai_score_unscaled'] = score_current[score_col[2:-3]].mean(axis=1)
        score_current['ai_score2_unscaled'] = score_current[score_col[2:-4]+['esg']].mean(axis=1)
        score_col += ['ai_score_unscaled', 'ai_score2_unscaled']

        # 1. test rank
        c1 = score_current.groupby(['currency_code'])['ai_score'].rank(axis=0).corr(score_current.groupby(['currency_code'])['ai_score_unscaled'].rank(axis=0))
        c2 = score_current['ai_score2'].rank(axis=0).corr(score_current['ai_score2_unscaled'].rank(axis=0))
        if (c1<0.9) or (c2<0.9):
            raise ValueError("================ ERROR: ai_score before & after scaler correlation is < 0.9 ==================")

        # 1. save descriptive csv
        save_topn_ticker(score_current)
        save_description(score_current)

        # 2. save descriptive plot
        plot_dist_score(score_current, 'current', score_col)
        plot_minmax_factor(pillar_current)

    # def test_corr(self):
    #
    #     score_history_current = score_history.loc[score_history['period_end']==score_history['period_end'].max()]
    #     score_history_current[score_col_history] = score_history_current.groupby(['currency_code'])[score_col_history].rank()
    #     score_current[score_col_current] = score_current.groupby(['currency_code'])[score_col_current].rank()
    #     score_history_current = score_history_current.merge(score_current, on=['ticker'], suffixes=('_test','_prod'))
    #     x = score_history_current.corr()

def save_topn_ticker(df, n=25):
    ''' save stock details for top 25 in each score '''

    df = df.loc[df['currency_code'].isin(['HKD','USD'])]

    writer = pd.ExcelWriter(f'#{suffixes}_ai_score_top{n}.xlsx')

    for i in ['wts_rating', 'dlp_1m', 'ai_score', 'ai_score2']:
        idx = df.groupby(['currency_code'])[i].nlargest(n).index.get_level_values(1)
        ddf = df.loc[idx].sort_values(by=['currency_code',i], ascending=False)
        ddf[['currency_code','ticker',i]].to_excel(writer, sheet_name=i, index=False)
        try:
            all_df.append(ddf)
        except:
            all_df = ddf.copy()

    all_df.drop_duplicates().to_excel(writer, sheet_name='original_scores', index=False)
    writer.save()

def save_description(df):
    ''' write statistics for  '''

    df = df.groupby(['currency_code']).agg(['min','mean', 'median', 'max', 'std','count']).transpose()
    print(df)
    df.to_csv(f'#{suffixes}_describe_current.csv')

def save_description_history(df):
    ''' write statistics for description '''

    df = df.groupby(['currency_code','period_end'])['ai_score'].agg(['min','mean', 'median', 'max', 'std','count'])
    print(df)
    df.to_csv(f'#{suffixes}_describe_history.csv')

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
        plt.close(fig)

def qcut_eval(score_col, fundamentals, name=''):
    ''' evaluate score history with 1) descirbe, 2) score 10-qcut mean ret, 3) per period change '''

    writer = pd.ExcelWriter(f'#{suffixes}_score_eval_history_{name}.xlsx')

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
    score_eval().test_current()
    # score_eval().test_history()
