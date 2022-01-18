from general.sql_process import read_query
import pandas as pd
import numpy as np
import global_vars

def update_top_universe_rating():
    ''' calculate weekly return for weekly pick of top score '''

    # start_date = '2021-11-01'
    # # start_date = '2022-01-01'
    #
    # # get top 10 pick for each currency each day
    # df = read_query(f"SELECT u.ticker, trading_day, field, value, currency_code FROM universe_rating_history r "
    #                     f"INNER JOIN (SELECT ticker, currency_code FROM universe WHERE is_active) u ON r.ticker=u.ticker "
    #                     f"WHERE trading_day >= '{start_date}' "
    #                     f"  AND currency_code in ('HKD', 'USD', 'EUR') "        # only audit on HKD, USD, EUR
    #                     f"  AND extract(dow from trading_day)=1 "               # Monday's Pick
    #                     , global_vars.db_url_alibaba_prod)
    # df["trading_day"] = pd.to_datetime(df["trading_day"])
    # df = df.pivot(index=["trading_day", "ticker", "currency_code"], columns=["field"], values="value").reset_index()
    #
    # # calculate weekly return from TRI
    # tri = read_query(f"SELECT ticker, trading_day, total_return_index FROM data_tri "
    #                      f"WHERE trading_day >= '{start_date}'"
    #                      , global_vars.db_url_alibaba_prod)
    # tri['trading_day'] = pd.to_datetime(tri['trading_day'])
    # tri = tri.pivot(index=["trading_day"], columns=["ticker"], values="total_return_index")
    # tri = tri.resample('D').pad().ffill()
    # triw = (tri.shift(-7)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "ret_week"})
    # trim = (tri.shift(-28)/tri - 1).stack().dropna(how="all").reset_index().rename(columns={0: "ret_month"})
    #
    # # map return to top picks
    # df = df.merge(triw, on=["ticker", "trading_day"], how="left")
    # df = df.merge(trim, on=["ticker", "trading_day"], how="left")
    #
    # df.to_csv('universe_rating_check.csv', index=False)

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

    for name, g in qcut_ret_mean.groupby(["currency_code", "index"]):
        print(name)
        print(g)

    return


if __name__ == '__main__':
    update_top_universe_rating()