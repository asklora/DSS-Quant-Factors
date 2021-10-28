import numpy as np
import pandas as pd
import global_vals
from utils_des import read_item_df

def best_factor():
    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM des_factor_trial_quantile2 WHERE years=5 and testing_interval=91 and pillar is not null ORDER BY score', conn)
        formula = pd.read_sql('SELECT name, pillar FROM factor_formula_ratios_descriptive', conn).set_index('name')['pillar'].to_dict()
    global_vals.engine_ali.dispose()

    df['pillar'] = df['pillar'].str.replace('avg_','')
    df['pillar'] = df['pillar'].str.replace('change_','')
    df['pillar'] = df['pillar'].replace(formula)

    for testing_interval, g in df.groupby('testing_interval'):
        groupby_col = ['dimension', 'n_cluster', 'method', 'pillar']
        g['rank'] = g.groupby(groupby_col)['score'].rank()
        avg = g.groupby(['pillar','cols','preprocess','dimension'])['rank'].mean().reset_index()
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

    cols1 = 'avg_debt_to_asset,avg_roe,avg_div_yield,change_earnings,change_tri_fillna,avg_volume'
    cols1 = cols1.split(',')
    x1 = data.org_x(cols1)

    cols2 = 'avg_earnings_yield,avg_ni_to_cfo,avg_ebitda_to_ev,avg_book_to_price'
    cols2 = cols2.split(',')
    x2, _ = data.pca_x(cols2, 2)

    df = pd.DataFrame(np.concatenate([x1, x2], axis=1), columns=cols1+['value_pca1', 'value_pca2'])
    c = df.corr().unstack().reset_index().drop_duplicates().sort_values(by=[0])
    print(c)

if __name__=="__main__":
    # best_factor()
    test_selection_score()