import global_vars
import pandas as pd
from collections import Counter
import lightgbm as lgb
from hierarchy_ratio_cluster import trim_outlier_std
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import scale

def read_fund_port(n=10):
    ''' Get top 5 holdings for mid-large size fund '''

    with global_vals.engine.connect() as conn, global_vals.engine_ali.connect() as conn_ali:
        uni = pd.read_sql(f"SELECT * FROM universe WHERE currency_code='USD'", conn)
        port = pd.read_sql('SELECT * FROM data_factor_eikon_fund_holdings', conn_ali)
        size = pd.read_sql('SELECT * FROM data_factor_eikon_fund_size ORDER BY tna', conn_ali)
    global_vals.engine.dispose()

    # filter fund with size in middle-low range (10% - 50%)
    size = size.loc[(size['tna']>size['tna'].quantile(0.1))&(size['tna']<size['tna'].quantile(0.5))]
    port = port.loc[port['fund'].isin(size['ric'].to_list())]
    port = port.drop_duplicates(subset=['fund','ticker'])

    # filter ticker in our universe
    port['ticker'] = port['ticker'].str.replace('.OQ', '.O')
    df = port.merge(uni, on=['ticker'], how='inner')

    # first 10 holdings
    valid_fund = (df.groupby('fund')['ticker'].count()>n)
    valid_fund = valid_fund.loc[valid_fund]
    df = df.loc[df['fund'].isin(list(valid_fund.index))]
    df = df.groupby('fund')[['fund', 'ticker']].head(n)

    f = Counter(df['fund'].to_list())
    print(f)
    t = Counter(df['ticker'].to_list())
    print(t)

    return df

def read_item_df(testing_interval=7):
    item_df = pd.read_csv(f'dcache_sample_{testing_interval}.csv')
    item_df = item_df.loc[item_df['trading_day']==item_df['trading_day'].max()]
    item_df = item_df.loc[item_df['ticker'].str[0]!='.']
    item_df.iloc[:, 2:] = trim_outlier_std(item_df.iloc[:, 2:].fillna(0)).values

    return item_df

def cv_portfolio(testing_interval=7, num_stock=5):

    user_df = read_fund_port(num_stock)
    item_df = read_item_df(testing_interval)

    df = pd.merge(user_df, item_df, on=['ticker'], how='left')
    cols = item_df.columns.to_list()[2:]

    params = {
        'objective':'regression',
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'max_bin': 256,
        'num_leaves': 50,
        'min_data_in_leaf': 2,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'min_gain_to_split': 0,
        'lambda_l1': 0,
        'lambda_l2': 10,
    }

    for col in cols:
        for i in range(num_stock):
            train_y = df.groupby('fund').nth(i)
            train_X = df.groupby('fund').apply(lambda x: x.reset_index(drop=True).drop(index=[i]).mean())

            train_X = scale(train_X[cols].values)
            train_y = train_y[col].values
            lgb_train = lgb.Dataset(train_X, label=train_y, free_raw_data=False)

            eval_hist = lgb.cv(params, lgb_train, num_boost_round=1000, feature_name=cols, nfold=5, eval_train_metric=True)
            print(eval_hist)

if __name__=="__main__":
    cv_portfolio()