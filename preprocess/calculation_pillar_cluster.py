import numpy as np
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import Counter
from sqlalchemy.dialects.postgresql import DATE, TEXT, INTEGER, JSON

import global_vars
from general.sql_process import read_query, upsert_data_to_database

from sklearn.preprocessing import scale
from sklearn.cluster import FeatureAgglomeration

dtype_pillar = dict(
    pillar=TEXT,
    factor_name=JSON,
    testing_period=DATE,
    weeks_to_expire=INTEGER,
    group=TEXT,
    subpillar_trh=INTEGER,
    pillar_trh=INTEGER,
    lookback=INTEGER
)


def calc_pillar_cluster(testing_period, weeks_to_expire, group='USD', subpillar_trh=5, pillar_trh=2, lookback=5,
                        save_to_db=True):
    """

    Parameters
    ----------
    testing_period (Datetime): testing_period
    weeks_to_expire (Int): weeks_to_expire
    group (Str): currency_code
    subpillar_trh (Int):
        nth smallest distances below which factors belong to same sub-pillar (i.e. factors won't be selected together)
    pillar_trh (Int):
        nth largest distances below which factors belong to same pillar
    lookback (Int):
        number of years for sample lookback periods prior to testing_period for clustering

    Returns
    -------

    """

    # download past [5] year ratio for clustering
    end_date = testing_period + relativedelta(weeks=weeks_to_expire)
    start_date = end_date - relativedelta(years=lookback)

    conditions = [f"ticker in (SELECT ticker FROM universe WHERE currency_code='{group}')",
                  f"trading_day <= '{end_date}'",
                  f"trading_day > '{start_date}'"]
    query = f"SELECT * FROM {global_vars.processed_ratio_table} WHERE {' AND '.join(conditions)}"
    df = read_query(query)
    # df.to_pickle('cache_factor_ratio1.pkl')
    # df = pd.read_pickle('cache_factor_ratio1.pkl')

    # get active factor list
    df_formula = read_query(f"SELECT name FROM {global_vars.formula_factors_table_prod} WHERE is_active")
    df_active_factor = df_formula['name'].to_list()

    # pivot ratio table & filter by active factor list
    df['trading_day'] = pd.to_datetime(df['trading_day'])
    df = df.set_index(['trading_day', 'ticker', 'field'])['value'].unstack()
    df = df.filter(df_active_factor)
    df = df.fillna(0)
    feature_names = np.array(df.columns.to_list())

    # general cluster for distance calculation
    X = scale(df)
    agglo = FeatureAgglomeration(n_clusters=None, distance_threshold=0, linkage='average')
    agglo.fit(X)

    # find [subpillar]
    subpillar_dist = agglo.distances_[subpillar_trh]
    subpillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=subpillar_dist, linkage='average')
    subpillar_cluster.fit(X)
    subpillar_label = subpillar_cluster.labels_
    print(Counter(subpillar_label))
    subpillar = {f"subpillar_{k}": list(feature_names[subpillar_label == k]) for k, v in
                 dict(Counter(subpillar_label)).items()
                 if (v < len(feature_names) * subpillar_trh) and (v > 1)}

    # find [pillar]
    pillar_dist = agglo.distances_[-pillar_trh]
    pillar_cluster = FeatureAgglomeration(n_clusters=None, distance_threshold=pillar_dist, linkage='average')
    pillar_cluster.fit(X)
    pillar_label = pillar_cluster.labels_
    print(Counter(pillar_label))
    pillar = {f"pillar_{k}": list(feature_names[pillar_label == k]) for k, v in dict(Counter(pillar_label)).items()}

    if save_to_db:
        config = dict(testing_period=testing_period, weeks_to_expire=weeks_to_expire,
                      group=group, subpillar_trh=subpillar_trh, pillar_trh=pillar_trh, lookback=lookback)
        save_pillar_to_db(subpillar, pillar, config)

    return subpillar, pillar


def save_pillar_to_db(subpillar, pillar, config):
    """ save pillar factor to DB """

    df_pillar_all = pd.DataFrame({"factor_name": {**pillar, **subpillar}})
    df_pillar_all = df_pillar_all.reset_index().rename(columns={"index": "pillar"})

    for k, v in config.items():
        df_pillar_all[k] = v

    primary_key = list(config.keys()) + ['pillar']
    upsert_data_to_database(df_pillar_all, global_vars.factors_pillar_cluster_table, primary_key=primary_key,
                            how="update", dtype=dtype_pillar)


if __name__ == '__main__':
    subpillar_dict, pillar_dict = calc_pillar_cluster(dt.date(2016, 1, 1), 4)