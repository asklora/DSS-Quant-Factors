import logging

import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import itertools
from general.report_to_slack import to_slack

from global_vars import *
from general.sql_process import read_query, upsert_data_to_database, trucncate_table_in_database, uid_maker

from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION
from sqlalchemy.sql.sqltypes import BOOLEAN

icb_num = 6

# define dtypes for final_member_df & final_results_df when writing to DB
results_dtypes = dict(
    group=TEXT,
    trading_day=DATE,
    factor_name=TEXT,
    stock_return_y=DOUBLE_PRECISION,
    trim_outlier=BOOLEAN
)

def trim_outlier(df, prc=0):
    ''' assign a max value for the 99% percentile to replace inf'''

    df_nan = df.replace([np.inf, -np.inf], np.nan)
    pmax = df_nan.quantile(q=(1 - prc))
    pmin = df_nan.quantile(q=prc)
    df = df.mask(df > pmax, pmax)
    df = df.mask(df < pmin, pmin)

    return df

def insert_prem_for_group(*args):
    ''' calculate premium for each group / factor and insert to Table [factor_processed_premium] '''

    def qcut(series):
        try:
            series_fillinf = series.replace([-np.inf, np.inf], np.nan)
            nonnull_count = series_fillinf.notnull().sum()
            if nonnull_count < 3:
                return series.map(lambda _: np.nan)
            elif nonnull_count > 65:
                prc = [0, 0.2, 0.8, 1]
            else:
                prc = [0, 0.3, 0.7, 1]
            
            q = pd.qcut(series_fillinf, prc, duplicates='drop')

            n_cat = len(q.cat.categories)
            if n_cat == 3:
                q.cat.categories = range(3)
            elif n_cat == 2:
                q.cat.categories = [0, 2]
            else:
                return series.map(lambda _: np.nan)
            q[series_fillinf == np.inf] = 2
            q[series_fillinf == -np.inf] = 0
            return q
        except ValueError as e:
            logging.debug(f'Premium not calculated: {e}')
            return series.map(lambda _: np.nan)

    df, group, factor, trim_outlier_, y_col = args
    weeks_to_expire, average_days = y_col.split('_')[-2][1:].astype(int), y_col.split('_')[-1][1:].astype(int)

    df = df.loc[df['currency_code']==group]     # Select all ticker for certain currency
    logging.info(f'=== Calculate premium for ({group}, {factor}) ===')

    try:
        df = df[['trading_day', y_col, factor]].dropna(how='any')
        if len(df) == 0:
            raise Exception(f"Either stock_return_y or ticker in group '{group}' is all missing")

        if trim_outlier_:
            df[y_col] = trim_outlier(df[y_col], prc=.05)

        df['quantile_group'] = df.groupby(['trading_day'])[factor].transform(qcut)
        df = df.dropna(subset=['quantile_group']).copy()
        df['quantile_group'] = df['quantile_group'].astype(int)
        prem = df.groupby(['trading_day', 'quantile_group'])[y_col].mean().unstack()

        # Calculate small minus big
        prem = (prem[0] - prem[2]).dropna().rename('value').reset_index()
        prem['group'] = group
        prem['field'] = factor
        prem['weeks_to_expire'] = weeks_to_expire
        if trim_outlier_:
            prem['field'] = 'trim_'+prem['field']
        prem = uid_maker(prem, primary_key=['group','trading_day','field','weeks_to_expire'])

        upsert_data_to_database(data=prem.sort_values(by=['group', 'trading_day']),
                                table=factor_premium_table,
                                primary_key=["uid"],
                                db_url=db_url_write,
                                how="update", verbose=-1)
    except Exception as e:
        to_slack("clair").message_to_slack(f"*[ERROR] in Calculate Premium*: {e}")
        return False
    return True

def calc_premium_all(weeks_to_expire, average_days=1, trim_outlier_=False, processes=12, all_groups=['USD','EUR'], start_date=None):
    '''  calculate factor premium for different configurations and write to DB Table [factor_premium_table]

    Parameters
    ----------
    weeks_to_expire (Int):
        forward period for premium calculation
    average_days (Int):
        number of average days for the stock returns used to calculate premiums
    trim_outlier_ (Bool, Optional):
        if True, use trimmed returns for top/bottom stocks
    processes (Int, Optional):
        multiprocess threads (default=12)
    all_groups (List[Str], Optional):
        currencies to calculate premiums (default=[USD, EUR])
    start_date (Date, Optional):
        start_date for premium calculation (default=None, i.e. calculate entire history)
    '''

    logging.info(f'=== Get {formula_factors_table_prod} ===')
    formula_query = f"SELECT * FROM {formula_factors_table_prod} WHERE is_active AND NOT(keep) "
    formula = read_query(formula_query, db_url_read)
    factor_list = formula['name'].to_list()  # factor = all variabales

    # premium calculate currency only
    ratio_query = f"SELECT r.*, u.currency_code " \
                  f"FROM {processed_ratio_table} r " \
                  f"INNER JOIN universe u ON r.ticker=u.ticker " \
                  f"WHERE currency_code in {tuple(all_groups)}"
    if start_date:
        ratio_query += f" AND trading_day>='{start_date}' "
    df = read_query(ratio_query.replace(",)",")"), db_url_read)
    df = df.loc[~df['ticker'].str.startswith('.')].copy()
    df = df.pivot(index=["ticker","trading_day", "currency_code"], columns=["field"], values='value').reset_index()
    y_col = f'stock_return_y_w{weeks_to_expire}_d{average_days}'
    df = df.dropna(subset=[y_col, 'ticker'])

    # resample df to match the weeks_to_expire
    date_list = reversed(df["trading_day"].unique())
    date_list = [x for i, x in enumerate(date_list) if (i % weeks_to_expire == 0)]
    df = df.loc[df["trading_day"].isin(date_list)]

    logging.info(f'Groups: {" -> ".join(all_groups)}')
    logging.info(f'trim_outlier: {trim_outlier_}')
    logging.info(f'Save to [{factor_premium_table}]')

    all_groups = itertools.product([df], all_groups, factor_list, [trim_outlier_], [y_col])
    all_groups = [tuple(e) for e in all_groups]

    # trucncate_table_in_database(f"{factor_premium_table}", db_url_write)
    with mp.Pool(processes=processes) as pool:
        pool.starmap(insert_prem_for_group, all_groups)

    to_slack("clair").message_to_slack(f"===  FINISH [update] DB [{factor_premium_table}] ===")

if __name__ == "__main__":

    last_update = datetime.now()
    # tbl_suffix_extra = ''

    start = datetime.now()

    calc_premium_all(weeks_to_expire=26, average_days=28, trim_outlier_=False, processes=1, start_date='2022-01-01')

    end = datetime.now()

    logging.debug(f'Time elapsed: {(end - start).total_seconds():.2f} s')
    # write_local_csv_to_db()
