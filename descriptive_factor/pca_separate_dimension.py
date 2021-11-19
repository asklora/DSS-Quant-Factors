import global_vars
import pandas as pd
import datetime as dt
from utils_des import read_item_df, feature_hierarchical_plot, selection_feature_hierarchical, \
    cluster_fcm, cluster_gaussian, cluster_hierarchical, report_to_slack, plot_scatter_2d

def define_pillars(df):
    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT pillar, name FROM {global_vals.formula_factors_table}_descriptive', conn)
    global_vals.engine_ali.dispose()

    lst = []
    for i in ['momentum','value','growth','efficiency']:
        org_name = formula.loc[formula['pillar']==i, 'name'].to_list()
        org_name = org_name + ['avg_' + x for x in org_name] + ['change_' + x for x in org_name]
        lst.append(list(set(org_name) & set(df.columns.to_list())))
    return lst

class test_factor:
    def __init__(self, suffixes='', start=5, end=0, testing_interval=91):
        self.suffixes = suffixes
        self.data = read_item_df(testing_interval=testing_interval)
        self.data.time_after(start, end)
        self.info = {'years': start, 'testing_interval': testing_interval}
    
        df = self.data.item_df
        df = df.drop(columns=['avg_mkt_cap','icb_code'])        # columns will be evaluate separately
        # feature_hierarchical_plot(df.iloc[:, 2:])
        # cols = selection_feature_hierarchical(df.iloc[:, 2:])
        self.cols = define_pillars(df)

    def try_pca(self):
        self.info['preprocess'] = 'pca'
        cluster_method = [cluster_fcm, cluster_gaussian, cluster_hierarchical]
        for pillar_cols in self.cols:
            self.info['pillar'] = pillar_cols[0]
            for n in [2, 3]:
                for i in [10, 20]:
                    X, explained_ratio = self.data.pca_x(cols=pillar_cols, n=n)
                    results = []
                    for func in cluster_method:
                        try:
                            score, y = func(X, n_clusters=i)
                            self.info1 = self.info.copy()
                            self.info1.update({'cols':','.join(pillar_cols), 'dimension': n, 'n_cluster':i, 'score': score,
                                         'method':func.__name__, 'explained_ratio': explained_ratio[-1]})
                            results.append(self.info1)
                        except Exception as e:
                            report_to_slack(e)
                    # print(pd.DataFrame(results))
                    with global_vals.engine_ali.connect() as conn:
                        pd.DataFrame(results).to_sql(f'des_factor_trial{self.suffixes}', conn, index=False, if_exists='append')
                    global_vals.engine_ali.dispose()

    def try_svd(self):
        self.info['preprocess'] = 'svd'
        cluster_method = [cluster_fcm, cluster_gaussian, cluster_hierarchical]
        for pillar_cols in self.cols:
            self.info['pillar'] = pillar_cols[0]
            for n in [2, 3]:
                for i in [10, 20]:
                    results = []
                    X, explained_ratio = self.data.svd_x(cols=pillar_cols, n=n)
                    for func in cluster_method:
                        # try:
                        score, y = func(X, n_clusters=i)
                        self.info = self.info.copy()
                        self.info.update({'cols':','.join(pillar_cols), 'dimension': n, 'n_cluster':i, 'score': score,
                                     'method': func.__name__, 'explained_ratio': explained_ratio[-1]})
                        results.append(self.info)
                        # except Exception as e:
                        #     report_to_slack(e)
                    # print(pd.DataFrame(results))
                    with global_vals.engine_ali.connect() as conn:
                        pd.DataFrame(results).to_sql(f'des_factor_trial{self.suffixes}', conn, index=False, if_exists='append')
                    global_vals.engine_ali.dispose()

    def try_original(self):
        self.info['preprocess'] = 'original'
        cluster_method = [cluster_hierarchical]
        self.info['pillar'] = 'all'
        pillar_cols = [i for x in self.cols[1:] for i in x]
        for n in [3]:
            comb_list = self.data.select_comb_x(pillar_cols, n_cols=n)
            for i in [5, 20]:        # n_clusters
                for col in comb_list:
                    print(col)
                    X = self.data.org_x(cols=list(col))
                    results = []
                    for func in cluster_method:
                        try:
                            score, y = func(X, n_clusters=i)
                            self.info1 = self.info.copy()
                            self.info1.update({'cols':','.join(col), 'dimension': n, 'n_cluster':i, 'score': score,
                                         'method':func.__name__})
                            results.append(self.info1)
                        except Exception as e:
                            report_to_slack(e)
                    with global_vals.engine_ali.connect() as conn:
                        pd.DataFrame(results).to_sql(f'des_factor_trial{self.suffixes}', conn, index=False, if_exists='append')
                    global_vals.engine_ali.dispose()

def plot_test_factor():

    method = cluster_gaussian
    cols = 'avg_interest_to_earnings,avg_debt_to_asset,avg_cash_ratio,avg_inv_turnover_re,avg_ca_turnover_re,avg_roe,avg_roic,avg_fa_turnover_re'
    cols = 'avg_ebitda_to_ev,avg_ni_to_cfo,avg_earnings_yield,avg_book_to_price'
    cols = 'avg_div_yield,avg_div_payout'
    cols = 'avg_inv_turnover_re,avg_cash_ratio'
    cols = cols.split(',')

    data = read_item_df(testing_interval=91, plot=True)
    data.time_after(5, 0)

    # df = data.item_df
    # df = df.drop(columns=['avg_market_cap_usd', 'icb_code'])  # columns will be evaluate separately
    # pillars = define_pillars(df)
    # for col in pillars:
    #     print(col)

    mom=['vol', 'avg_volume', 'skew', 'change_volume', 'change_tri_fillna', 'avg_volume_1w3m']
    val=['avg_earnings_yield', 'avg_ni_to_cfo', 'avg_book_to_price', 'avg_ebitda_to_ev']
    grow=['avg_capex_to_dda', 'change_revenue', 'change_earnings', 'change_assets', 'avg_div_yield', 'change_ebtda', 'avg_div_payout', 'avg_gross_margin']
    eff=['avg_fa_turnover_re', 'avg_roe', 'avg_interest_to_earnings', 'avg_ca_turnover_re', 'avg_cash_ratio', 'avg_roic', 'avg_debt_to_asset', 'avg_inv_turnover_re']

    lst = []
    # for col in data.orginal_cols:
    # for col in [mom, val, grow, eff]:
    for n_clusters in [20]:
        X = data.org_x(cols=cols)
        # X, explained_ratio = data.pca_x(cols=col, n=1)
        score, y = method(X, n_clusters=n_clusters)
        lst.append({'name':cols, 'score':score, 'n_cluster': n_clusters})
        print(cols, score, n_clusters)

    r = pd.DataFrame(lst).sort_values('score')
    r.to_csv(f'results_{dt.datetime.now()}.csv')
    print(r)
    plot_scatter_2d(X, y)

def check_corr():
    data = read_item_df(testing_interval=91, plot=False)
    data.time_after(5, 0)
    df = data.item_df
    c = df.corr().unstack().reset_index().drop_duplicates().sort_values(by=[0])
    print(c)

def test_grid_search(testing_interval, start):

    data = read_item_df(testing_interval=testing_interval)
    data.time_after(start, 0)
    info = {'years': start, 'testing_interval': testing_interval}

    df = data.item_df
    df = df.drop(columns=['avg_mkt_cap', 'icb_code'])  # columns will be evaluate separately
    cols = define_pillars(df)

    cluster_method = [cluster_hierarchical]
    pillar_cols = [i for x in cols[1:] for i in x]
    for n in [3]:
        comb_list = data.select_comb_x(pillar_cols, n_cols=n)
        for col in comb_list:
            info1 = info.copy()
            info1.update({'cols': ','.join(col), 'dimension': n})
            for i in [5, 10, 20]:  # n_clusters
                print(col)
                X = data.org_x(cols=list(col))
                results = []
                for func in cluster_method:
                    try:
                        score, y = func(X, n_clusters=i)
                        info1[f'cluster_{i}'] = score
                        results.append(info1)
                    except Exception as e:
                        report_to_slack(e)
            with global_vals.engine_ali.connect() as conn:
                pd.DataFrame(results).to_sql(f'des_factor_trial_original3', conn, index=False, if_exists='append')
            global_vals.engine_ali.dispose()

if __name__=="__main__":
    # test_factor(suffixes='_quantile3').try_pca()        # 2: right for selection / 3: start combine pillars
    # test_factor(suffixes='_quantile3').try_svd()
    for y, t in [[5, 91],[1, 7]]:
        print(y,t)
        test_grid_search(start=y, testing_interval=t)

    # plot_test_factor()
    # check_corr()