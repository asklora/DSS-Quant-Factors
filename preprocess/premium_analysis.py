import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn import linear_model

from global_vars import *
from general.sql_process import read_table, read_query

def download_premium():
    ''' download premium and calculate factor importance '''

    query = f"SELECT * FROM {factor_premium_table}"
    df = read_query(query, db_url_read)

    for i in [1, 3, 5, 10]:
        start_date = (dt.datetime.today() - relativedelta(years=i)).date()
        df_period = df.loc[df["trading_day"] >= start_date]
        df_period_avg = df_period.groupby(["field", "group", "weeks_to_expire"])["value"].mean().unstack(level=[-2, -1])
        df_period_avg = df_period_avg.sort_values(by=[("USD", 1)], ascending=False)

        df_period["value"] = df_period["value"].abs()
        df_period_abs_avg = df_period.groupby(["field", "group", "weeks_to_expire"])["value"].mean().unstack(level=[-2, -1])
        df_period_abs_avg = df_period_abs_avg.sort_values(by=[("USD", 1)], ascending=False)
        print(df_period_avg)


if __name__ == '__main__':
    download_premium()