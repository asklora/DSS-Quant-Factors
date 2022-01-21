from general.sql_process import read_query
import pandas as pd
import numpy as np
from global_vars import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def get_top_picks(start_date = '2021-11-01'):
    ''' match Monday Scores with next week/month return '''

    # get top 10 pick for each currency each day
    df = read_query(f"SELECT u.ticker, trading_day, field, value, currency_code FROM universe_rating_history r "
                        f"INNER JOIN (SELECT ticker, currency_code FROM universe WHERE is_active) u ON r.ticker=u.ticker "
                        f"WHERE trading_day >= '{start_date}' "
                        f"  AND currency_code in ('HKD', 'USD', 'EUR') "        # only audit on HKD, USD, EUR
                        f"  AND extract(dow from trading_day)=1 "               # Monday's Pick
                        , db_url_alibaba_prod)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df = df.pivot(index=["trading_day", "ticker", "currency_code"], columns=["field"], values="value").reset_index()

    # calculate weekly return from TRI
    tri = read_query(f"SELECT ticker, trading_day, total_return_index FROM data_tri "
                         f"WHERE trading_day >= '{start_date}'"
                         , db_url_alibaba_prod)
    tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    tri = tri.pivot(index=["trading_day"], columns=["ticker"], values="total_return_index")
    tri = tri.resample('D').pad().ffill()
    triw = (tri.shift(-7)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "ret_week"})
    trim = (tri.shift(-28)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "ret_month"})

    # map return to top picks
    df = df.merge(triw, on=["ticker", "trading_day"], how="left")
    df = df.merge(trim, on=["ticker", "trading_day"], how="left")

    df.to_csv('universe_rating_check.csv', index=False)

def check_average_score_return_by_industry():
    ''' calculate avg weekly return for weekly pick of top score
    -----
    Results:
    1. USD (travel & leisure) always high
    '''

    df = pd.read_csv('universe_rating_check.csv')
    industry = read_query("SELECT ticker, name_4 as industry_name FROM universe u INNER JOIN icb_code_explanation e "
                          "ON u.industry_code::text=e.code_8 WHERE is_active", db_url_alibaba_prod)

    df = df.merge(industry, on=["ticker"], how="left")

    score_col = ['ai_score_unscaled',
        'fundamentals_extra_monthly1', 'fundamentals_extra_weekly1', 'fundamentals_momentum_monthly1',
        'fundamentals_momentum_weekly1', 'fundamentals_quality_monthly1', 'fundamentals_quality_weekly1',
        'fundamentals_value_monthly1', 'fundamentals_value_weekly1', 'penalty', 'wts_rating']

    avg = df.groupby(["currency_code", "industry_name"])[score_col+["ret_week", "ret_month"]].mean()
    avg = avg.reset_index().sort_values(by=["ai_score_unscaled"], ascending=False)

    for name, g in avg.groupby(["currency_code"]):
        print(name)
        print(g)

def check_avg_return_by_score_qcut():
    ''' calculate avg weekly return for weekly pick of top score '''

    df = pd.read_csv('universe_rating_check.csv')

    num_col = df.select_dtypes(float).columns.to_list()
    print(num_col)

    score_col = ['ai_score_unscaled',
        'fundamentals_extra_monthly1', 'fundamentals_extra_weekly1', 'fundamentals_momentum_monthly1',
        'fundamentals_momentum_weekly1', 'fundamentals_quality_monthly1', 'fundamentals_quality_weekly1',
        'fundamentals_value_monthly1', 'fundamentals_value_weekly1', 'penalty', 'wts_rating']

    def group_qcut_ret(g):
        g_q = g[score_col].apply(pd.qcut, q=10, labels=False, duplicates="drop")
        g_ret = g[["ticker", "ret_week", "ret_month"]]
        q_ret = []
        for col in score_col:
            g_ret["q"] = g_q[col].values
            ret = g_ret.groupby("q").mean().transpose().reset_index()
            ret["score"] = col
            q_ret.append(ret)
        q_ret = pd.concat(q_ret, axis=0)
        return q_ret

    qcut_ret = df.groupby(["currency_code", "trading_day"]).apply(group_qcut_ret).reset_index().drop(columns=["level_2"])
    qcut_ret_mean = qcut_ret.groupby(["currency_code", "index", "score"]).mean().reset_index()
    qcut_ret_mean["diff"] = qcut_ret_mean[9] - qcut_ret_mean[0]
    qcut_ret_mean = qcut_ret_mean.sort_values(by=["diff"], ascending=False)

    qcut_ret_momentum = qcut_ret.loc[(qcut_ret["score"]=="fundamentals_momentum_weekly1") &
                                     (qcut_ret["index"] == "ret_week") &
                                     (qcut_ret["currency_code"] == "USD")
                                     ].sort_values(by=["trading_day"], ascending=False)
    qcut_ret_momentum = qcut_ret_momentum.set_index(["trading_day"])[list(range(10))]
    sns.heatmap(scale(qcut_ret_momentum.values.T).T)
    plt.show()

    for name, g in qcut_ret_mean.groupby(["currency_code", "index"]):
        print(name)
        print(g)

if __name__ == '__main__':
    # get_top_picks()
    check_avg_return_by_score_qcut()
    # check_average_score_return_by_industry()