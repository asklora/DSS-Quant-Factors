import global_vars
import pandas as pd
import datetime as dt
import numpy as np
from utils_des import read_item_df, feature_hierarchical_plot, selection_feature_hierarchical, \
    cluster_fcm, cluster_gaussian, cluster_hierarchical, report_to_slack, plot_scatter_2d, plot_scatter_nd

class test_91_factor:
    def __init__(self, suffixes='', start=5, end=0):
        self.suffixes = suffixes
        self.data = read_item_df(testing_interval=91)
        self.data.time_after(start, end)
        self.info = {'years': start, 'testing_interval': 91}

        cols1 = 'avg_volume,change_volume,'
        cols1 += 'avg_mkt_cap,'
        cols1 += 'industry_code'
        cols1 = cols1.split(',')
        x1 = self.data.org_x(cols1)

        cols2 = 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price'
        cols2 = cols2.split(',')
        x2, _ = self.data.pca_x(cols2, 2)

        cols3 = 'avg_roic,avg_debt_to_asset,avg_ca_turnover_re,avg_cash_ratio,avg_roe,avg_inv_turnover_re,avg_fa_turnover_re,avg_interest_to_earnings'
        cols3 = cols3.split(',')
        x3, _ = self.data.pca_x(cols3, 2)

        cols4 = 'change_earnings,change_ebtda,change_assets,avg_gross_margin,avg_div_yield,avg_capex_to_dda,change_revenue,avg_div_payout'
        cols4 = cols4.split(',')
        x4, _ = self.data.svd_x(cols4, 2)

        self.info['pillar'] = 'best4_everything'
        self.all_cols = cols1 + ['value_pca1', 'value_pca2'] + ['effi_svd1', 'effi_svd2'] + ['grow_pca1', 'grow_pca2']
        self.df = pd.DataFrame(np.concatenate([x1, x2, x3, x4], axis=1), columns=self.all_cols)

    def try_original(self):
        cluster_method = [cluster_fcm, cluster_gaussian, cluster_hierarchical]
        for n in [2, 3]:
            comb_list = self.data.select_comb_x(self.all_cols, n_cols=n)
            for i in [10, 20]:  # n_clusters
                for col in comb_list:
                    print(col)
                    X = self.df[list(col)].values
                    results = []
                    for func in cluster_method:
                        score, y = func(X, n_clusters=i)
                        self.info1 = self.info.copy()
                        self.info1.update(
                            {'cols': ','.join(col), 'dimension': n, 'n_cluster': i, 'score': score,
                             'method': func.__name__})
                        results.append(self.info1)

                    with global_vars.engine_ali.connect() as conn:
                        pd.DataFrame(results).to_sql(f'des_factor_trial{self.suffixes}', conn, index=False,
                                                     if_exists='append')
                    global_vars.engine_ali.dispose()

class test_7_factor:
    def __init__(self, suffixes='', start=1, end=0):
        self.suffixes = suffixes
        self.data = read_item_df(testing_interval=7)
        self.data.time_after(start, end)
        self.info = {'years': start, 'testing_interval': 7}

        cols1 = 'change_volume,avg_volume,avg_volume_1w3m,'
        cols1 += 'avg_mkt_cap,'
        cols1 += 'industry_code'
        cols1 = cols1.split(',')
        x1 = self.data.org_x(cols1)

        cols2 = 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price'
        cols2 = cols2.split(',')
        x2, _ = self.data.pca_x(cols2, 2)

        cols3 = 'avg_roe,avg_cash_ratio'
        cols3 = cols3.split(',')
        x3 = self.data.org_x(cols3)

        cols4 = 'avg_div_payout,avg_div_yield'
        cols4 = cols4.split(',')
        x4 = self.data.org_x(cols4)

        self.info['pillar'] = 'best4_everything'
        self.all_cols = cols1 + ['value_pca1', 'value_pca2'] + cols3 + cols4
        self.df = pd.DataFrame(np.concatenate([x1, x2, x3, x4], axis=1), columns=self.all_cols)

        c = self.df.corr().unstack().reset_index().drop_duplicates(subset=[0]).sort_values(by=[0], ascending=False)
        print(c)

    def try_original(self):
        cluster_method = [cluster_fcm, cluster_gaussian, cluster_hierarchical]
        for n in [2, 3]:
            comb_list = self.data.select_comb_x(self.all_cols, n_cols=n)
            for i in [10, 20]:  # n_clusters
                for col in comb_list:
                    print(col)
                    X = self.df[list(col)].values
                    results = []
                    for func in cluster_method:
                        score, y = func(X, n_clusters=i)
                        self.info1 = self.info.copy()
                        self.info1.update(
                            {'cols': ','.join(col), 'dimension': n, 'n_cluster': i, 'score': score,
                             'method': func.__name__})
                        results.append(self.info1)

                    with global_vars.engine_ali.connect() as conn:
                        pd.DataFrame(results).to_sql(f'des_factor_trial{self.suffixes}', conn, index=False,
                                                     if_exists='append')
                    global_vars.engine_ali.dispose()

def test_icb_mkt(testing_interval=91, years=5):

    from dateutil.relativedelta import relativedelta
    item_df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')
    item_df = item_df.sort_values(by=['ticker', 'trading_day'])
    item_df = item_df.loc[item_df['ticker'].str[0] != '.']
    item_df = item_df.loc[item_df['ticker'].str[-3:] != '.HK']

    item_df['trading_day'] = pd.to_datetime(item_df['trading_day'])
    item_df = item_df.loc[item_df['trading_day'] >= (dt.datetime.now() - relativedelta(years=years))]
    item_df = item_df.loc[item_df['trading_day'] <= (dt.datetime.now() - relativedelta(years=0))]

    cols = 'change_ebtda,avg_div_yield,avg_div_payout,'
    cols += 'avg_cash_ratio,avg_roe,avg_interest_to_earnings'

    cols = cols.strip(',').split(',')
    c = item_df[cols].corr().unstack().reset_index().drop_duplicates(subset=[0]).sort_values(by=[0])
    c.to_csv('correlation_all.csv')
    exit(1)

    from sklearn.preprocessing import quantile_transform, scale
    item_df['avg_mkt_cap'] = quantile_transform(item_df[['avg_mkt_cap']], output_distribution='normal', n_quantiles=1000)[:,0]
    X = item_df[['industry_code', 'avg_mkt_cap']].dropna(how='any').values

    import scipy
    def string_match_distance(u, v):
        d = 8
        u, v = str(int(u[0])), str(int(v[0]))
        for i in range(len(u)):
            if u[i]==v[i]:
                d-=1
            else:
                break
        return d

    # icb = item_df['industry_code'].astype(int).astype(str).str.split('',expand=True).iloc[:,1:-1]
    mkt = scipy.spatial.distance.pdist(X[:, [-1]], 'euclidean')
    mkt = scipy.spatial.distance.squareform(mkt)
    icb = scipy.spatial.distance.pdist(X[:, [0]], string_match_distance)
    icb = scipy.spatial.distance.squareform(icb)*np.max(mkt)/8
    X = icb/2 + mkt/2

    # X = scipy.spatial.distance.pdist(X, 'euclidean')
    # X = scipy.spatial.distance.squareform(X)

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    result = []
    for n in [5, 10, 20]:
        kwargs = {'distance_threshold': None, 'linkage': 'complete', 'n_clusters': n, 'affinity': 'precomputed'}
        model = AgglomerativeClustering(**kwargs).fit(X)
        y = model.labels_
        score = silhouette_score(X, y, metric="precomputed")
        result.append(score)
        print(result)

def test_case(suffixes, testing_interval=91, years=5):
    data = read_item_df(testing_interval=testing_interval)
    data.time_after(years, 0)
    info = {'years': years, 'testing_interval': testing_interval}

    cols = ''
    # cols += 'avg_mkt_cap,'
    # cols += 'avg_fa_turnover_re,avg_interest_to_earnings,avg_roe,avg_ca_turnover_re,avg_cash_ratio,avg_roic,avg_inv_turnover_re,avg_debt_to_asset,'
    # cols += 'change_ebtda,avg_div_payout,avg_gross_margin,change_revenue,avg_div_yield,change_assets,change_earnings,avg_capex_to_dda,'
    # cols += 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price,'
    # cols += 'avg_volume,skew,change_tri_fillna,change_volume,avg_volume_1w3m,vol'
    # cols +='avg_div_yield,avg_div_payout,'
    # cols +='avg_debt_to_asset,avg_cash_ratio,'
    # cols +='avg_volume,change_volume,avg_volume_1w3m,'
    # cols +='vol,change_tri_fillna,avg_volume'
    # cols +='avg_book_to_price,avg_earnings_yield,change_revenue'
    # cols += 'industry_code,'
    cols += 'avg_div_yield,avg_inv_turnover_re,avg_cash_ratio'

    cols = cols.strip(',').split(',')
    n_pca = None       #TODO
    if n_pca:
        X,_ = data.pca_x(cols, n_pca)
    else:
        X = data.org_x(cols)

    cluster_method = [cluster_gaussian, cluster_hierarchical]
    results = []
    for i in [5, 10, 20]:  # n_clusters
        info1 = info.copy()
        info1.update({'cols': ','.join(cols), 'dimension': len(cols), 'n_cluster': i, 'n_pca': n_pca})
        for func in cluster_method:
            score, y = func(X, n_clusters=i)
            info1.update({func.__name__: score})
            # plot_scatter_2d(X, y)
            plot_scatter_nd(X, y, cols)
            continue
        results.append(info1)

    # with global_vars.engine_ali.connect() as conn:
    #     pd.DataFrame(results).to_sql(f'des_factor_trial{suffixes}', conn, index=False, if_exists='append')
    # global_vars.engine_ali.dispose()

if __name__ == "__main__":
    test_91_factor(suffixes='_quantile3').try_original()
    # test_7_factor(suffixes='_quantile4').try_original()
    # test_case('_quantile6')
    # test_icb_mkt()
