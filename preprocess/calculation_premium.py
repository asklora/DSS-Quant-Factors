import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import itertools

import global_vars
from utils_sql import sql_read_query, upsert_data_to_database, trucncate_table_in_database

from sqlalchemy.dialects.postgresql import DATE, TEXT, DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.sql.sqltypes import BOOLEAN, TIMESTAMP

icb_num = 6

# define dtypes for final_member_df & final_results_df when writing to DB
results_dtypes = dict(
    group=TEXT,
    period_end=DATE,
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
            print(e)
            return series.map(lambda _: np.nan)

    df, group, tbl_suffix, factor, trim_outlier_ = args
    print(group, tbl_suffix, factor, trim_outlier_)

    try:
        df = df[['period_end','stock_return_y', factor]].dropna(how='any')
        if len(df) == 0:
            raise Exception(f"Either stock_return_y or ticker in group '{group}' is all missing")

        if trim_outlier_:
            df['stock_return_y'] = trim_outlier(df['stock_return_y'], prc=.05)
            tbl_suffix_extra = '_v2_trim'
        else:
            tbl_suffix_extra = '_v2'

        if factor in df.columns.to_list():
            df['quantile_group'] = df.groupby(['period_end'])[factor].transform(qcut)
            df = df.dropna(subset=['quantile_group']).copy()
            df['quantile_group'] = df['quantile_group'].astype(int)
            prem = df.groupby(['period_end', 'quantile_group'])['stock_return_y'].mean().unstack()

            # Calculate small minus big
            prem = (prem[0] - prem[2]).dropna().rename('premium').reset_index()
            prem['group'] = group
            prem['factor_name'] = factor
            prem['trim_outlier'] = trim_outlier_

            upsert_data_to_database(data=prem.sort_values(by=['group', 'period_end']),
                                    table=f"{global_vars.factor_premium_table}{tbl_suffix}{tbl_suffix_extra}",
                                    primary_key=["group", "period_end", "factor_name", "trim_outlier"],
                                    db_url=global_vars.db_url_alibaba_prod,
                                    how="append")
    except Exception as e:
        print(e)
        return False

    return True

def calc_premium_all_v2(tbl_suffix, trim_outlier_=False, processes=12, all_groups=['USD']):

    ''' calculate factor premium for different configurations:
        1. monthly sample + using last day price
        2. biweekly sample + using last day price
        3. monthly sample + using average price of last week
    '''

    # Read stock_return / ratio table
    print(f'#################################################################################################')
    print(f'      ------------------------> Download ratio data from DB')
    if trim_outlier_:
        tbl_suffix_extra = '_v2_trim'
    else:
        tbl_suffix_extra = '_v2'

    formula_query = f"SELECT * FROM {global_vars.formula_factors_table_prod} WHERE is_active"
    formula = sql_read_query(formula_query, global_vars.db_url_alibaba_prod)
    factor_list = formula['name'].to_list()  # factor = all variabales

    # permium calculate USD only
    ratio_query = f"SELECT * FROM {global_vars.processed_ratio_table}{tbl_suffix} WHERE currency_code = 'USD'"
    df = sql_read_query(ratio_query, global_vars.db_url_alibaba_prod)
    df = df.dropna(subset=['stock_return_y', 'ticker'])
    df = df.loc[~df['ticker'].str.startswith('.')].copy()

    print(f'      ------------------------> Groups: {" -> ".join(all_groups)}')
    print(f'      ------------------------> Save to {global_vars.factor_premium_table}{tbl_suffix}{tbl_suffix_extra}')

    all_groups = itertools.product([df], all_groups, [tbl_suffix], factor_list, [trim_outlier_])
    all_groups = [tuple(e) for e in all_groups]

    trucncate_table_in_database(f"{global_vars.factor_premium_table}{tbl_suffix}{tbl_suffix_extra}",
                                global_vars.db_url_alibaba_prod)
    with mp.Pool(processes=processes) as pool:
        res = pool.starmap(insert_prem_for_group, all_groups)

    return res


if __name__ == "__main__":

    last_update = datetime.now()
    # tbl_suffix_extra = '_v2'

    start = datetime.now()

    calc_premium_all_v2(tbl_suffix='_weekly1', trim_outlier_=False, processes=6)

    end = datetime.now()

    print(f'Time elapsed: {(end - start).total_seconds():.2f} s')
    # write_local_csv_to_db()
