from sqlalchemy import text
from sqlalchemy.dialects.postgresql.base import DATE, TEXT, INTEGER, BOOLEAN, TIMESTAMP
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd

import global_vals

stock_pred_dtypes = dict(
    period_end=DATE,
    factor_name=TEXT,
    group=TEXT,
    factor_weight=INTEGER,
    long_large=BOOLEAN,
    last_update=TIMESTAMP
)

def download_stock_pred(q, iter_name, rank_along_testing_history=True, keep_last_period=True, return_summary=False):
    ''' organize production / last period prediction and write weight to DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type, P.group as group_code, S.neg_factor, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_rf_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' and tree_type ='rf' and use_pca = 0.2 ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number'], keep='last')

    if keep_last_period:
        result_all = result_all.loc[result_all['testing_period']==result_all['testing_period'].max()]   # keep only last testing i.e. for production

    neg_factor = result_all[['group_code','neg_factor']].drop_duplicates().set_index('group_code')['neg_factor'].to_dict()

    # use average predictions from different validation sets
    result_all = result_all.groupby(['testing_period','y_type','group_code'])['pred'].mean().unstack()

    if rank_along_testing_history:
        result_all = result_all.reset_index()
        result_all.columns.name = None
        result_all['testing_period'] = pd.to_datetime(result_all['testing_period'])
        result_all = result_all.rename(columns={'testing_period': 'period_end', 'y_type': 'factor_name'})

        result_all = result_all.melt(id_vars=['period_end', 'factor_name'], value_vars=result_all.select_dtypes('float').columns, var_name='group', value_name='pred')
        result_all['factor_weight'] = result_all.groupby('group')['pred'].transform(lambda x: pd.qcut(x, q=[0., 1/3, 2/3, 1.], labels=range(3), duplicates='drop'))
        
        if return_summary:
            factor_rank = result_all.pivot(['period_end'], ['factor_name', 'group'], ['factor_weight']).droplevel(0, axis=1).sort_index(axis=1)
            rank_count = result_all[['period_end', 'group', 'factor_weight']].value_counts()
            rank_count = rank_count.reset_index().pivot(['period_end'], ['group', 'factor_weight'])
            rank_count = rank_count.fillna(0).droplevel(0, axis=1).sort_index(axis=1)
        
        if keep_last_period:
            result_all = result_all.loc[result_all['period_end']==result_all['period_end'].max(), ['period_end','factor_name','group','factor_weight']].reset_index(drop=True).copy()
    else:
        # classify predictions to n-bins
        result_all = result_all.apply(pd.qcut, q=q, labels=False).stack().dropna().reset_index()
        # set format of the sql table
        result_all.columns = ['period_end','factor_name','group','factor_weight']

    result_all = result_all.dropna(axis=0, subset=['factor_weight'])

    # result_all['factor_name'] = result_all['factor_name'].str[2:]
    result_all['period_end'] = result_all['period_end'] + MonthEnd(1)
    result_all['factor_weight'] = result_all['factor_weight'].astype(int)
    result_all['long_large'] = False
    result_all['last_update'] = dt.datetime.now()

    for k, v in neg_factor.items():
        result_all.loc[(result_all['group']==k)&(result_all['factor_name'].isin([x[2:] for x in v.split(',')])), 'long_large'] = True

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000, 'dtype': stock_pred_dtypes}
        conn.execute(f"DELETE FROM {global_vals.production_factor_rank_table} "
                     f"WHERE period_end='{dt.datetime.strftime(result_all['period_end'][0], '%Y-%m-%d')}'")   # remove same period prediction if exists
        result_all.sort_values(['group','factor_weight']).to_sql(global_vals.production_factor_rank_table, **extra)
    global_vals.engine_ali.dispose()

    if return_summary:
        return factor_rank, rank_count

if __name__ == "__main__":
    download_stock_pred(3, 'pca_trimold2', rank_along_testing_history=True)

