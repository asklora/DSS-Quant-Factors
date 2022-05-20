import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vars
import re
import argparse
import datetime as dt
from dateutil.relativedelta import relativedelta
from general.send_slack import to_slack
from general.send_email import send_mail
from general.sql_process import read_query

suffixes = dt.datetime.today().strftime('%Y%m%d')

def backdate_by_week_day(week=0, day=0):
    return (dt.datetime.now().date() - relativedelta(weeks=week) - relativedelta(days=day)).strftime("%Y-%m-%d")

def get_weekly_return(tickers):
    ''' get weekly return for top picks '''

    tickers += [".SPX", ".HSI"]

    query = "SELECT ticker, trading_day, total_return_index as tri FROM master_ohlcvtr "
    query += f"WHERE trading_day > '{backdate_by_week_day(5,1)}' AND ticker in {tuple(tickers)} "
    price = read_query(query, global_vars.db_url_aws_read)

    from preprocess.calculation_ratio import fill_all_day
    price = fill_all_day(price).sort_values(by=["ticker", "trading_day"])
    dates = price["trading_day"].to_list()
    price["tri"] = price.groupby(["ticker"])["tri"].ffill()
    for i in range(4):
        price[f"{i+1}th-week Return ({dates[-(28-i*7)].strftime('%Y-%m-%d')})"] = \
            price[f"tri"].shift(28-7*i-7) / price.groupby(["ticker"])["tri"].shift(28-7*i) - 1

    return price.groupby(["ticker"]).last().drop(columns=["tri", "trading_day"])

def topn_ticker(n=20, DEBUG=False):
    ''' save stock details for top 25 in each score '''

    query = f"SELECT u.ticker, currency_code, trading_day, ticker_name, ai_score, company_description FROM universe_rating_history r "
    query += f"INNER JOIN (SELECT ticker, ticker_name, currency_code, company_description FROM universe) u ON r.ticker=u.ticker "
    query += f"WHERE trading_day in ('{backdate_by_week_day(0,1)}', '{backdate_by_week_day(1,1)}', '{backdate_by_week_day(2,1)}', '{backdate_by_week_day(3,1)}', '{backdate_by_week_day(4,1)}') "
    query += f"AND currency_code in ('HKD', 'USD') "
    df = read_query(query, global_vars.db_url_aws_read)

    df = df.sort_values(by="ai_score", ascending=False).groupby(by=["currency_code", "trading_day"]).head(20)
    price = get_weekly_return(df["ticker"].to_list())
    df = df.merge(price, left_on=["ticker"], right_index=True, how="outer")

    writer = pd.ExcelWriter(f'#{suffixes}_ai_score_top{n}.xlsx')
    for name, g in df.groupby(by=["trading_day"]):
        ddf = g.sort_values(by=["currency_code", "ai_score"], ascending=False).drop(columns=["trading_day"])
        ddf.to_excel(writer, sheet_name=name.strftime("%Y-%m-%d"), index=False)
    writer.save()

    to_slack().file_to_slack(f'#{suffixes}_ai_score_top{n}.xlsx', 'xlsx', f'Top {n} tickers')  # send to factor_message channel
    print(dt.datetime.today().weekday())
    if (dt.datetime.today().weekday() == 0) or DEBUG:  # on Monday send all TOP Picks
        subject = "Weekly Top 20 Pick (HKD & USD)"
        if DEBUG:
            subject = "(Resent) " + subject
        text = "The attached Excel includes top 20 tickers with the highest ai score for the past 4 week (include this week). " \
               "This email will be automatically send out every Monday morning. " \
               "Feel free to contact Clair if there is any issue."
        file = f"#{suffixes}_ai_score_top{n}.xlsx"
        send_mail(subject, text, file, "clair.cui@loratechai.com")
        send_mail(subject, text, file, "stepchoi@loratechai.com")
        send_mail(subject, text, file, "nick.choi@loratechai.com")
        send_mail(subject, text, file, "john.kim@loratechai.com")
        send_mail(subject, text, file, "kenson.lau@loratechai.com")
        send_mail(subject, text, file, "joseph.chang@loratechai.com")
        send_mail(subject, text, file, "nickey.kong@loratechai.com")

class score_eval:
    def __init__(self, SLACK=False, currency=None, DEBUG=False):
        self.SLACK = SLACK
        self.currency = currency
        self.DEBUG = DEBUG

    def test_current(self):
        ''' test on ai_score current '''

        pillar_current = {}
        with global_vars.engine_ali.connect() as conn_ali, global_vars.engine.connect() as conn:
            # update_time = pd.read_sql(f'SELECT * FROM {global_vars.update_time_table}', conn_ali)
            # update_time['update_time'] = update_time['update_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            query = f"SELECT currency_code, S.* FROM {global_vars.production_score_current} S "
            query += f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker "
            if self.currency:
                query += f"WHERE currency_code='{self.currency}'"
            score_current = pd.read_sql(query, conn)
            query1 = f"SELECT currency_code, S.* FROM {global_vars.production_score_current_history} S "
            query1 += f"INNER JOIN (SELECT ticker, currency_code FROM universe) U ON S.ticker=U.ticker "
            # query1 += f"WHERE currency_code='{self.currency}'"
            score_current_history = pd.read_sql(query1, conn)
            for p in filter(None, score_current['currency_code'].unique()):
                for i in ['_weekly1','_monthly1']:
                    try:
                        pillar_current[(p,i)] = pd.read_sql(f"SELECT * FROM test_fundamental_score_details_{p}{i}", conn_ali)
                    except Exception as e:
                        to_slack("clair").message_to_slack(e)
        global_vars.engine_ali.dispose()
        global_vars.engine.dispose()

        # if self.SLACK:
        #     report_series_to_slack('*======== Tables Update Time ========*', update_time.set_index('index')['update_time'])

        # 1. save comparison csv
        self.__compare_to_history(score_current, score_current_history)

        # 2. save correlation before/after scaling
        self.__corr_scale_unscale(score_current)

        # 3. save descriptive csv
        # self.__save_topn_ticker(score_current)
        self.__save_description(score_current)

        # 4. save descriptive plot
        self.__plot_dist_score(score_current, 'current-DLPA')
        self.__plot_dist_score(score_current, 'current-fundamentals')
        self.__plot_dist_score(score_current, 'current-final')
        self.__plot_minmax_factor(pillar_current)

    def __compare_to_history(self, current, history):
        ''' compare current score -> history score '''
        score_current_history = history.loc[history['trading_day'] < history['trading_day'].max()]
        score_history_last = score_current_history.loc[score_current_history['trading_day'] == score_current_history['trading_day'].max()]
        score_history_avg = score_current_history.groupby(['ticker']).mean().reset_index()
        lw_comp_des, lw_comp = self.__save_compare(current, score_history_last)
        avg_comp_des, avg_comp = self.__save_compare(current, score_history_avg)

        writer = pd.ExcelWriter(f'#{suffixes}_compare.xlsx')
        lw_comp_des.to_excel(writer, sheet_name='Average Score Change (last bday)')
        lw_comp.to_excel(writer, sheet_name='Top Score Change (last bday)')
        avg_comp_des.to_excel(writer, sheet_name='Average Score Change (history average)')
        avg_comp.to_excel(writer, sheet_name='Top Score Change (history average)')
        writer.save()

        if self.SLACK:
            to_slack().series_to_slack('*======== Compare with Last Business Day (Mean Change) ========*', lw_comp_des['mean'])
            to_slack().series_to_slack('*======== Compare with Score History Average (Mean Change) ========*', avg_comp_des['mean'])
            to_slack().file_to_slack(f'./#{suffixes}_compare.xlsx', 'xlsx', f'Compare score')

    def __corr_scale_unscale(self, current):
        ''' compare scaled score -> unscaled score '''
        c1 = current.groupby(['currency_code'])['ai_score'].rank(axis=0).corr(current.groupby(['currency_code'])['ai_score_unscaled'].rank(axis=0))
        c2 = current['ai_score2'].rank(axis=0).corr(current['ai_score2_unscaled'].rank(axis=0))
        if self.SLACK:
            to_slack(). message_to_slack(f'======== ai_score before & after scaler correlation: {round(c1, 3)} ========')
            to_slack(). message_to_slack(f'======== ai_score2 before & after scaler correlation: {round(c2, 3)} ========')

    def __save_compare(self, score_cur, score_his):
        cols = ['wts_rating', 'dlp_1m', 'ai_score', 'ai_score2']
        df = score_cur.set_index('ticker').dropna(subset=['ai_score']).merge(score_his.set_index('ticker').dropna(subset=['ai_score']),
                                                left_index=True, right_index=True, how='inner', suffixes=('_cur','_his'))

        # score changes
        arr = df[[x+'_cur' for x in cols]].values - df[[x+'_his' for x in cols]].values
        r = pd.DataFrame(arr, columns=cols, index=df.index).sort_values(by='ai_score', ascending=True)
        r_des = r.describe().transpose()
        r_top = pd.concat([r.sort_values(by='ai_score').head(20), r.sort_values(by='ai_score').tail(20)], axis=0)
        return r_des, r_top

    def __save_description(self, df):
        ''' write statistics for  '''

        df = df.groupby(['currency_code']).agg(['min','mean', 'median', 'max', 'std','count']).transpose()
        print(df)

        writer = pd.ExcelWriter(f'#{suffixes}_describe_current.xlsx')
        df.to_excel(writer, sheet_name='Distribution Current')
        writer.save()

        if self.SLACK:
            for col in ['ai_score','ai_score_unscaled','ai_score2','ai_score2_unscaled']:
                df_save = df.xs(col, level=0).round(2)
                to_slack().df_to_slack(f'*======== Score Distribution - {col} ========*', df_save)
            to_slack().file_to_slack(f'#{suffixes}_describe_current.xlsx', 'xlsx', f'{global_vars.production_score_current} Score Distribution')

    def __plot_dist_score(self, df, filename):
        ''' Plot distribution (currency, score)  for all AI score compositions '''

        num_cur = len(df['currency_code'].unique())
        score_col = df.select_dtypes(float).columns.to_list()
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
        to_slack("clair").file_to_slack(f'#{suffixes}_score_dist_{filename}.png', 'png', f'{filename} Score Distribution')

    def __plot_minmax_factor(self, df_dict):
        ''' plot min/max distribution '''

        for (cur, freq), g in df_dict.items():

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

            fig_name = cur+freq
            plt.suptitle(fig_name, fontsize=30)
            fig.savefig(f'#{suffixes}_score_minmax_{fig_name}.png')
            to_slack("clair").file_to_slack(f'#{suffixes}_score_minmax_{fig_name}.png', 'png', f'{fig_name} Score Detailed Distribution')
            plt.close(fig)

def read_query(query, engine_num=int):
    ''' read table from different DB '''
    engine_dict = {0: global_vars.engine, 1: global_vars.engine_ali, 2: global_vars.engine_ali_prod}
    with engine_dict[engine_num].connect() as conn:
        df = pd.read_sql(query, conn)
    engine_dict[engine_num].dispose()
    return df

if __name__ == "__main__":
    topn_ticker(20, DEBUG=True)
    exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--slack', action='store_true', help='Send message/file to Slack = True')
    parser.add_argument('--debug', action='store_true', help='Debug = True')
    parser.add_argument('--currency', default='HKD')
    args = parser.parse_args()
    print(args)

    topn_ticker(n=20, DEBUG=args.debug)
    eval = score_eval(SLACK=args.slack, currency=args.currency, DEBUG=args.debug)
    eval.test_current()     # test on universe_rating + test_fundamentals_score_details_{currency}
