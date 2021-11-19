import global_vars
import pandas as pd
import numpy as np
import datetime as dt
from utils_des import read_item_df, feature_hierarchical_plot, selection_feature_hierarchical, \
    cluster_fcm, cluster_gaussian, cluster_hierarchical, report_to_slack, plot_scatter_2d

def test_tech(testing_interval=91, sample_interval=5, n_factor=3, n_clusters=20):
    ''' test technical factor results '''

    info = {'pillar': 'tech'}

    # 1. each currency
    for cur in ['HKD', 'USD']:

        # 2. each time
        for start in np.arange(sample_interval, 21, sample_interval):

            data = read_item_df(testing_interval=testing_interval, currency=cur)
            info['currency'] = cur
            info['testing_interval'] = testing_interval

            data.time_after(start, start-sample_interval)
            info['start_year'] = start
            info['sample_interval'] = sample_interval

            # 3. each factor combintions
            info['n_factor'] = n_factor
            info['n_clusters'] = n_clusters
            cols = 'avg_volume,skew,change_tri_fillna,change_volume,avg_volume_1w3m,vol'
            cols = cols.strip(',').split(',')

            # 3.1. using PCA
            info['cols'] = 'PCA'
            X, explained_ratio = data.pca_x(cols=cols, n=n_factor)
            score, y = cluster_hierarchical(X, n_clusters=n_clusters)
            info['score'] = score
            cluster_write_to_sql(info)

            # 3.2. using heuristic
            if testing_interval==7:
                cols = 'avg_volume,change_tri_fillna,vol'
                cols = cols.strip(',').split(',')

            for c in data.select_comb_x(cols, n_cols=n_factor):
                info['cols'] = ', '.join(list(c))
                X = data.org_x(cols=list(c))
                score, y = cluster_hierarchical(X, n_clusters=n_clusters)
                info['score'] = score
                cluster_write_to_sql(info)


def test_funda(testing_interval=91, sample_interval=5, n_factor=3, n_clusters=20):
    ''' test technical factor results '''

    info = {'pillar': 'funda'}

    # 1. each currency
    for cur in ['HKD', 'USD']:

        # 2. each time
        for start in np.arange(sample_interval, 21, sample_interval):

            data = read_item_df(testing_interval=testing_interval, currency=cur)
            info['currency'] = cur
            info['testing_interval'] = testing_interval

            data.time_after(start, start - sample_interval)
            info['start_year'] = start
            info['sample_interval'] = sample_interval

            # 3. each factor combintions
            info['n_factor'] = n_factor
            info['n_clusters'] = n_clusters
            cols = 'avg_fa_turnover_re,avg_interest_to_earnings,avg_roe,avg_ca_turnover_re,avg_cash_ratio,avg_roic,avg_inv_turnover_re,avg_debt_to_asset,'
            cols += 'change_ebtda,avg_div_payout,avg_gross_margin,change_revenue,avg_div_yield,change_assets,change_earnings,avg_capex_to_dda,'
            cols += 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price,'
            cols = cols.strip(',').split(',')

            all = data.org_x(cols=list(cols))
            total_var = np.var(all, axis=0).sum()

            # 3.1. using PCA
            info['cols'] = 'PCA'

            X, explained_ratio, components = data.pca_x(cols=cols, n=None)
            components1 = components*explained_ratio
            print(components1)
            s1 = components.sum(axis=1)
            s2 = s1.sum()

            X, explained_ratio,_ = data.pca_x(cols=cols, n=n_factor)
            print(testing_interval, cur, explained_ratio)
            score, y = cluster_hierarchical(X, n_clusters=n_clusters)
            info['score'] = score

            cluster_write_to_sql(info)

            # 3.2. using heuristic
            cols_comb = [
                # 'avg_div_yield,avg_cash_ratio,avg_fa_turnover_re',
                # 'avg_debt_to_asset,avg_inv_turnover_re,avg_cash_ratio',
                # 'avg_earnings_yield,avg_div_yield,avg_cash_ratio',
                # 'avg_div_yield,avg_gross_margin,avg_cash_ratio',
                # 'avg_div_yield,avg_cash_ratio,avg_fa_turnover_re,avg_earnings_yield,',
                # 'avg_div_yield,avg_cash_ratio,avg_fa_turnover_re,avg_earnings_yield,avg_debt_to_asset',
                'avg_div_yield,avg_cash_ratio,avg_fa_turnover_re,avg_earnings_yield,avg_debt_to_asset,avg_gross_margin',
            ]

            for cols in cols_comb:
                cols = cols.strip(',').split(',')
                info['cols'] = ', '.join(list(cols))
                X = data.org_x(cols=list(cols))

                subset_var = np.var(X, axis=0).sum()
                print('explained_ratio: ', subset_var/total_var)

                score, y = cluster_hierarchical(X, n_clusters=n_clusters)
                info['score'] = score
                cluster_write_to_sql(info)

def cluster_write_to_sql(r):
    ''' write cluster results to SQL '''

    print(f"Org cols: {r['cols']}, n_cluster: {r['n_clusters']} ----> {r['score']}")
    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append'}
        pd.DataFrame(r, index=[0]).to_sql(f"des_factor_final", **extra)
    global_vals.engine_ali.dispose()

if __name__=="__main__":

    # test_tech(testing_interval=91, sample_interval=5, n_factor=2, n_clusters=20)
    # test_tech(testing_interval=7, sample_interval=1, n_factor=3, n_clusters=20)

    # test_funda(testing_interval=91, sample_interval=5, n_factor=4, n_clusters=10)
    test_funda(testing_interval=7, sample_interval=1, n_factor=3, n_clusters=20)