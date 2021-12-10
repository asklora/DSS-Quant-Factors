import numpy as np
import pandas as pd
import global_vars
from utils_des import read_item_df

def best_factor():
    with global_vars.engine_ali.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM des_factor_trial_original3 WHERE testing_interval=91', conn)
        formula = pd.read_sql('SELECT name, pillar FROM factor_formula_ratios_descriptive', conn).set_index('name')['pillar'].to_dict()
    global_vars.engine_ali.dispose()

    # df['preprocess'] = 'NA'
    # df['pillar'] = df['pillar'].str.replace('avg_','')
    # df['pillar'] = df['pillar'].str.replace('change_','')
    # df['pillar'] = df['pillar'].replace(formula)
    # df = df.loc[df['preprocess']=='original']
    # df = df.loc[df['method']=='cluster_hierarchical']
    # df = df.loc[df['dimension']==3]

    # df = pd.pivot_table(df, index=['cols'], columns=['n_cluster'], values='score').reset_index()
    df['avg'] = df[['cluster_5','cluster_10','cluster_20']].mean(axis=1)
    print(df)
    exit(1)


    for testing_interval, g in df.groupby('testing_interval'):
        groupby_col = ['dimension', 'n_cluster', 'method', 'pillar']
        g['rank'] = g.groupby(groupby_col)['score'].rank()
        avg = g.groupby(['pillar','cols','preprocess','dimension'])['rank','score'].median().reset_index()
        avg = avg.sort_values(['pillar','rank']).groupby('pillar').head(5)
        print(avg)
        avg.to_csv(f'des_best_rank_{testing_interval}.csv')
    exit(1)
    best = df.groupby(groupby_col).first()[['preprocess','cols','score']].reset_index()
    for name, g in best.groupby(['testing_interval','method']):
        print(name)
        print(g)

def test_selection_score():
    data = read_item_df(testing_interval=91)
    data.time_after(5, 0)

    cols1 ='avg_mkt_cap,'
    cols1 +='industry_code,'
    # cols1 = 'avg_debt_to_asset,avg_roe,avg_div_yield,change_earnings,change_tri_fillna,avg_volume'
    # cols1 = 'avg_ca_turnover_re,avg_interest_to_earnings,'
    # cols1 +='change_ebtda,avg_div_yield,'
    # cols1 ='avg_debt_to_asset,avg_cash_ratio,'
    # cols1 +='avg_div_yield,avg_div_payout,'
    cols1 +='avg_volume,change_volume,avg_volume_1w3m,'
    cols1 = cols1.strip(',').split(',')
    x1 = data.org_x(cols1)
    # df = pd.DataFrame(np.concatenate([x1], axis=1), columns=cols1)
    # c = df.corr().unstack().reset_index().drop_duplicates().sort_values(by=[0])
    # print(c)

    cols2 = 'avg_fa_turnover_re,avg_interest_to_earnings,avg_roe,avg_ca_turnover_re,avg_cash_ratio,avg_roic,avg_inv_turnover_re,avg_debt_to_asset,'
    cols2 += 'change_ebtda,avg_div_payout,avg_gross_margin,change_revenue,avg_div_yield,change_assets,change_earnings,avg_capex_to_dda,'
    cols2 += 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price,'
    cols2 = cols2.strip(',').split(',')
    x2, _ = data.pca_x(cols2, 2)

    df = pd.DataFrame(np.concatenate([x1, x2], axis=1), columns=cols1+['value_pca1', 'value_pca2'])
    c = df.corr().unstack().reset_index().drop_duplicates(subset=[0]).sort_values(by=[0])
    print(c)

    cols3 = 'avg_roic,avg_debt_to_asset,avg_ca_turnover_re,avg_cash_ratio,avg_roe,avg_inv_turnover_re,avg_fa_turnover_re,avg_interest_to_earnings'
    cols3 = cols3.split(',')
    x3, _ = data.pca_x(cols3, 2)

    cols4 = 'change_earnings,change_ebtda,change_assets,avg_gross_margin,avg_div_yield,avg_capex_to_dda,change_revenue,avg_div_payout'
    cols4 = cols4.split(',')
    x4, _ = data.svd_x(cols4, 2)

    df = pd.DataFrame(np.concatenate([x1, x2, x3, x4], axis=1), columns=cols1+['value_pca1', 'value_pca2']+['effi_svd1', 'effi_svd2']+['grow_pca1', 'grow_pca2'])
    c = df.corr().unstack().reset_index().drop_duplicates(subset=[0]).sort_values(by=[0], ascending=False)
    print(c)

if __name__=="__main__":
    best_factor()
    # test_selection_score()