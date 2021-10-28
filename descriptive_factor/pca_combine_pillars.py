import global_vals
import pandas as pd
import datetime as dt
import numpy as np
from utils_des import read_item_df, feature_hierarchical_plot, selection_feature_hierarchical, \
    cluster_fcm, cluster_gaussian, cluster_hierarchical, report_to_slack, plot_scatter_2d

def define_pillars(df):
    with global_vals.engine_ali.connect() as conn:
        formula = pd.read_sql(f'SELECT pillar, name FROM {global_vals.formula_factors_table}_descriptive', conn)
    global_vals.engine_ali.dispose()

    lst = {}
    for i in ['momentum', 'value', 'growth', 'efficiency']:
        org_name = formula.loc[formula['pillar'] == i, 'name'].to_list()
        org_name = org_name + ['avg_' + x for x in org_name] + ['change_' + x for x in org_name]
        lst[i] = list(set(org_name) & set(df.columns.to_list()))
    return lst

class test_factor:
    def __init__(self, suffixes='', start=5, end=0):
        self.suffixes = suffixes
        self.data = read_item_df(testing_interval=91)
        self.data.time_after(start, end)
        self.info = {'years': start, 'testing_interval': 91}

        df = self.data.item_df
        df = df.drop(columns=['avg_mkt_cap', 'icb_code'])  # columns will be evaluate separately
        self.pillar_cols_dict = define_pillars(df)

        cols1 = 'avg_volume,change_volume,'
        cols1 += 'avg_mkt_cap,'
        cols1 += 'icb_code'
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

        self.info = 'best4_everything'
        self.all_cols = cols1 + ['value_pca1', 'value_pca2'] + ['effi_svd1', 'effi_svd2'] + ['grow_pca1', 'grow_pca2']
        self.df = pd.DataFrame(np.concatenate([x1, x2, x3, x4], axis=1), columns=self.all_cols)

    def try_original(self):
        cluster_method = [cluster_fcm, cluster_gaussian, cluster_hierarchical]
        for n in [2, 3]:
            comb_list = self.data.select_comb_x(self.all_cols, n_cols=n)
            for i in [10, 20]:  # n_clusters
                for col in comb_list:
                    print(col)
                    X = self.df[col]
                    results = []
                    for func in cluster_method:
                        score, y = func(X, n_clusters=i)
                        self.info1 = self.info.copy()
                        self.info1.update(
                            {'cols': ','.join(col), 'dimension': n, 'n_cluster': i, 'score': score,
                             'method': func.__name__})
                        results.append(self.info1)

                    with global_vals.engine_ali.connect() as conn:
                        pd.DataFrame(results).to_sql(f'des_factor_trial{self.suffixes}', conn, index=False,
                                                     if_exists='append')
                    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    test_factor(suffixes='_quantile3').try_original()
