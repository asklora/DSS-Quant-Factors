import global_vars
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from utils_sql import sql_read_query, sql_read_table
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from utils_des import cluster_hierarchical
from preprocess.calculation_ratio import fill_all_day
from collections import Counter
from sklearn.preprocessing import maxabs_scale

def get_items(testing_interval=91, start=1):
    from utils_des import read_item_df
    data = read_item_df(testing_interval=testing_interval, currency="HKD")
    data.time_after(start, 0)

    df = data.item_df
    df_org = data.item_df_org
    return df, df_org

def get_items_pillar(testing_interval, start):
    df, df_org = get_items(testing_interval, start)
    id_cols = ["ticker", "trading_day"]

    # pillar = {}
    # mom_cols = 'avg_volume,skew,change_tri_fillna,change_volume,avg_volume_1w3m,vol'
    # mom_cols = 'vol,change_tri_fillna,avg_volume'
    #
    # mom_cols = mom_cols.strip(',').split(',')
    # pillar["momentum"] = df[id_cols + mom_cols]
    # pillar["id"] = df[id_cols+['avg_mkt_cap', 'icb_code']]

    cols = 'avg_fa_turnover_re,avg_interest_to_earnings,avg_roe,avg_ca_turnover_re,avg_cash_ratio,avg_roic,avg_inv_turnover_re,avg_debt_to_asset,'
    cols += 'change_ebtda,avg_div_payout,avg_gross_margin,change_revenue,avg_div_yield,change_assets,change_earnings,avg_capex_to_dda,'
    cols += 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price,'
    cols = list(cols.strip(',').split(','))

    import matplotlib.pyplot as plt
    X = df[cols].values
    model = PCA(n_components=None).fit(X)  # calculation Cov matrix is embeded in PCA
    components = pd.DataFrame(model.components_, columns=cols).transpose()
    explained_ratio = np.cumsum(model.explained_variance_ratio_)
    plt.plot(explained_ratio)
    plt.savefig('PCA_sample.png')
    exit(29000)

    df[["fpca_1", "fpca_2", "fpca_3"]] = PCA(n_components=2).fit_transform(df[cols].values)
    pillar["fundamental"] = df[id_cols+["fpca_1", "fpca_2", "fpca_3"]]

    for name, g in pillar.items():

        from utils_des import cluster_hierarchical
        score, y = cluster_hierarchical(g.set_index(["ticker", "trading_day"]).values, n_clusters=20)
        g["cluster"] = y

        from preprocess.calculation_ratio import fill_all_day
        g = fill_all_day(g).sort_values(by=['ticker','trading_day'])
        num_col = g.select_dtypes(float).columns.to_list()
        g[num_col] = g[num_col].ffill()
        g["trading_day"] = g["trading_day"].dt.date
        pillar[name] = g

    return pillar

def get_rating_history():
    ''' get rating history '''
    query = "SELECT ticker, trading_day, ai_score FROM universe_rating_history"
    df = sql_read_query(query, global_vars.db_url_aws_read)
    return df

def get_rating():
    ''' get rating history '''
    query = "SELECT * FROM universe_rating_history"
    df = sql_read_query(query, global_vars.db_url_aws_read)
    return df

def get_user():
    ''' read user details from firestore '''

    # Get a database reference to our posts
    if not firebase_admin._apps:
        cred = credentials.Certificate(global_vars.firebase_url)
        default_app = firebase_admin.initialize_app(cred)

    db = firestore.client()
    doc_ref = db.collection(u"prod_portfolio").get()

    object_list = []
    for data in doc_ref:
        format_data = {}
        data = data.to_dict()
        format_data['user_id'] = data.get('user_id')
        format_data['email'] = data.get('profile').get('email')
        object_list.append(format_data)

    result = pd.DataFrame(object_list).sort_values(by=['user_id'], ascending=False)
    return result

from utils_des import read_item_df

class calc_recommend_score:
    ''' Give recommendation score given user trading history '''

    def __init__(self):

        self.id_cols = ["ticker", "trading_day"]

        # 1. get user transaction from orders_position
        op_query = "SELECT user_id, ticker, created, current_returns, event, updated, bot_id, exchange_rate*investment_amount as investment_amount FROM orders_position "
        op_query += f"WHERE created>'2021-09-30'"
        user = sql_read_query(op_query, global_vars.db_url_aws_read)
        user["trading_day"] = pd.to_datetime(user["created"].dt.date)
        user["investment_amount"] = (user["investment_amount"]/10000).round(0)

        # 1.1. calculate user holding period (weighted by investment amount?)
        user["holding_period"] = (user["updated"] - user["created"])/dt.timedelta(days=1)
        # x = user[["user_id","bot_id","ticker","holding_period", "created","updated","current_returns"]].sort_values(by=["user_id", "ticker"])
        user["holding_period"] = user["holding_period"].mul(user["investment_amount"], axis=0)
        user_holding = user.groupby(["user_id"])["holding_period"].sum() / user.groupby(["user_id"])["investment_amount"].sum()
        print(user_holding.mean())
        user_holding_pct = user_holding/np.minimum(user_holding.max(), 30)

        # import matplotlib.pyplot as plt
        # plt.hist(user_holding, bins=20)
        # plt.suptitle('non-weighted')
        # plt.savefig('user_holding_non-weight.png')
        # plt.close()
        # exit(200)

        # 2. get ticker universe (map ticker -> name)
        universe_query = "SELECT ticker_name, company_description, ticker FROM universe"
        universe = sql_read_query(universe_query, global_vars.db_url_aws_read)
        ticker_to_name = universe.set_index(["ticker"])["ticker_name"].to_dict()

        # 3. get items & merge with user
        period = {"long_term": (91, 5),
                  "short_term": (7, 1)
                  }

        pillar = {"tech": self._get_tech_pillar,
                  "funda": self._get_funda_pillar,
                  "id": self._get_id_pillar
                  }

        # 3.1. separate score for long / short term
        for term, (interval, start_year) in period.items():
            distance_list = []
            similarity_list = []

            # 3.2. separate score for each pillar
            for name, func in pillar.items():
                item = func(interval, start_year)
                item = self.__cluster(item, n=20)
                item_col = item.columns.to_list()[2:-1]
                user_item = user.merge(self.__fill_all_day(item), on=["ticker", "trading_day"], how="left").dropna(subset=["cluster"])

                # 3.1: calculate importance based on past trading variance with in each pillar
                var_df = self.__calc_var(user_item, item_col, ord=1)
                var_df.name = name
                distance_list.append(var_df)

                # import matplotlib.pyplot as plt
                # plt.hist(var_df, bins=100)
                # plt.suptitle(name)
                # plt.savefig(name+'_l1_weight')
                # plt.close()
                # continue

                # 3.2: find similar stocks based on cluster
                item["trading_day"] = item["trading_day"].dt.date
                item_recent = item.loc[item["trading_day"] == item["trading_day"].max(), ["ticker", "cluster"]]

                cluster_mode = user_item.groupby(["user_id", "cluster"])["ticker"].count().reset_index()
                cluster_mode["count"] = cluster_mode.groupby(["user_id"])["ticker"].transform("sum")
                cluster_mode["count"] = cluster_mode["ticker"] / cluster_mode["count"]
                similarity = item_recent.merge(cluster_mode[["user_id", "cluster", "count"]], on=["cluster"])
                similarity = similarity.rename(columns={"count": name}).set_index(["user_id","ticker"])[name]
                similarity_list.append(similarity)
                print(similarity)

            pillar_name = list(pillar.keys())
            distance = pd.concat(distance_list, axis=1)
            distance[pillar_name] = 1-maxabs_scale(distance[pillar_name])
            distance = distance.div(distance.sum(axis=1), axis=0)
            distance = distance.reset_index()

            similarity = pd.concat(similarity_list, axis=1).reset_index().fillna(0)
            similarity = similarity.merge(distance, on=["user_id"], how="outer", suffixes=('_d', '_s'))
            similarity["ticker_name"] = similarity["ticker"].map(ticker_to_name)

        print(distance)

    def __get_items(self, testing_interval=91, start=1):
        data = read_item_df(testing_interval=testing_interval, currency="HKD")
        data.time_after(start, 0)
        return data.item_df

    def _get_tech_pillar(self, testing_interval, start):
        ''' get dataframe for technical factors '''
        df = self.__get_items(testing_interval, start)
        # mom_cols = 'avg_volume,skew,change_tri_fillna,change_volume,avg_volume_1w3m,vol'
        mom_cols = 'vol,change_tri_fillna,avg_volume'
        mom_cols = mom_cols.strip(',').split(',')
        return df[self.id_cols + mom_cols]

    def _get_id_pillar(self, testing_interval, start):
        ''' get dataframe for ID factors '''
        df = self.__get_items(testing_interval, start)
        return df[self.id_cols + ['avg_mkt_cap', 'icb_code']]

    def _get_funda_pillar(self, testing_interval, start):
        ''' get dataframe for fundamental factors (after PCA) '''
        df = self.__get_items(testing_interval, start)
        cols = 'avg_fa_turnover_re,avg_interest_to_earnings,avg_roe,avg_ca_turnover_re,avg_cash_ratio,avg_roic,avg_inv_turnover_re,avg_debt_to_asset,'
        cols += 'change_ebtda,avg_div_payout,avg_gross_margin,change_revenue,avg_div_yield,change_assets,change_earnings,avg_capex_to_dda,'
        cols += 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price,'
        cols = cols.strip(',').split(',')
        df[["fpca_1", "fpca_2", "fpca_3"]] = PCA(n_components=3).fit_transform(df[cols].values)
        return df[self.id_cols + ["fpca_1", "fpca_2", "fpca_3"]]

    def __cluster(self, g, n):
        ''' hierarchical clustering on pillar tickers '''
        score, y = cluster_hierarchical(g.set_index(["ticker", "trading_day"]).values, n_clusters=n)
        g["cluster"] = y
        return g

    def __fill_all_day(self, result, date_col="trading_day"):
        ''' fill all days for pillar df to prepare for merge with user data '''

        # Construct indexes for all day between first/last day * all ticker used
        df = result[["ticker", date_col]].copy()
        df.trading_day = pd.to_datetime(df[date_col])
        result.trading_day = pd.to_datetime(result[date_col])
        df = df.sort_values(by=[date_col], ascending=True)
        daily = pd.date_range(df.iloc[0, 1], dt.datetime.today(), freq='D')
        indexes = pd.MultiIndex.from_product([df['ticker'].unique(), daily], names=['ticker', date_col])

        # Insert weekend/before first trading date to df
        df = df.set_index(['ticker', date_col]).reindex(indexes).reset_index()
        df = df.sort_values(by=['ticker', date_col], ascending=True)
        result = df.merge(result, how="left", on=["ticker", date_col])

        # Fill forward
        num_col = result.select_dtypes(float).columns.to_list()
        result[num_col] = result[num_col].ffill()
        return result

    def __calc_var(self, df, item_cols, ord=1):

        # Weighted 1: by investment amount
        df[item_cols] = df[item_cols].mul(df["investment_amount"], axis=0)

        # calculate variance (L1 or L2)
        item_cols_mid = [f'mid_{x}' for x in item_cols]
        df[item_cols_mid] = df.groupby(["user_id"])[item_cols].transform("mean")
        item_cols_norm = [f'norm_{x}' for x in item_cols]
        if ord==2:
            df[item_cols_norm] = (df[item_cols].values - df[item_cols_mid].values)**ord
        elif ord==1:
            df[item_cols_norm] = np.abs(df[item_cols].values - df[item_cols_mid].values)
        return df.groupby("user_id")[item_cols_norm].mean().mean(axis=1)

def old():

    universe = get_universe()
    ticker_to_name = universe.set_index(["ticker"])["ticker_name"].to_dict()
    des_df = df.groupby(by=["user_id", "ticker"]).sum().reset_index()
    des_df = des_df.merge(universe, on=["ticker"]).sort_values(by=["user_id", "final_pnl_amount"])

    user_var = pd.DataFrame()
    knn_tickers_all = []
    df = df.drop(columns=["final_pnl_amount"])
    df = df.drop_duplicates(subset=["ticker","trading_day","user_id"])

    for p, item in get_items_pillar(testing_interval, start=1).items():


        user_item = df.merge(item, on=["ticker", "trading_day"], how="left")

        var_df = user_item.groupby("user_id").var().dropna(how='any')
        user_var[(p, testing_interval)] = var_df.mean(axis=1)

        user_item = user_item.merge(user_ticker_pct, on=["ticker", "user_id"], how="left")



        # get similar stocks
        # item = item.sort_values(["trading_day", "ticker"])
        # latest_item = item.groupby("ticker").last().drop(columns=["trading_day"])
        # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(latest_item.values)
        # user_mean = X.groupby("user_id").mean()
        # distances, indices = nbrs.kneighbors(user_mean.values)
        # le = LabelEncoder().fit(list(latest_item.index))
        # knn_tickers = pd.DataFrame(indices).apply(le.inverse_transform)
        # knn_tickers.index = user_mean.index
        # knn_tickers["type"] = f"{p}_{testing_interval}"
        # print(knn_tickers)
        # knn_tickers_all.append(knn_tickers.reset_index())

    knn_tickers_all = pd.concat(knn_tickers_all, axis=0)
    for i in range(5):
        knn_tickers_all[i] = knn_tickers_all[i].map(ticker_to_name)

def check_factor_change():
    import matplotlib.pyplot as plt
    df, df_org = get_items(91, 5)
    df = df.set_index(["ticker", "trading_day"]).stack().reset_index()
    minmax = df.groupby(["ticker", "level_2"])[0].agg(["min","mean","max"])
    minmax["diff"] = minmax["max"] - minmax["min"]
    minmax = minmax.sort_values(by=['max'], ascending=False).head(20)
    df_org["trading_day"] = pd.to_datetime(df_org["trading_day"])
    minmax = minmax.reset_index()[["ticker","level_2"]]

    df = df.merge(minmax, on=["ticker","level_2"])
    print(minmax)
    # for ticker, factor in minmax:
    #     ddf = df_org.loc[(df_org['ticker']==ticker), ["trading_day", factor]].sort_values(by=["trading_day"])
    #     plt.plot(ddf.dropna(how='any').set_index("trading_day")[factor])
    #     print(ddf)
    #     # plt.suptitle(ticker)
    #     # plt.xlabel(factor)
    #     # plt.show()
    #     # continue
    #     plt.savefig(f'check_factor_{ticker}_{factor}.png')
    #     plt.close()
    # print(df)

if __name__=="__main__":
    # get_items_pillar(91, 5)
    # get_transaction()
    # calc_recommend_score()
    check_factor_change()
