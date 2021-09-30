import pandas as pd
import global_vals
from sqlalchemy.dialects.postgresql import TEXT
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
from dateutil.relativedelta import relativedelta

test_url = "postgres://loratech:loraTECH123@pgm-3ns7dw6lqemk36rgpo.pg.rds.aliyuncs.com:5432/postgres"
test_engine = create_engine(test_url, max_overflow=-1, isolation_level="AUTOCOMMIT")

with global_vals.engine.connect() as conn, test_engine.connect() as conn_test:  # write stock_pred for the best hyperopt records to sql
    df = pd.read_sql(f"SELECT ticker, trading_day, open, high, low, close, total_return_index FROM master_ohlcvtr "
                     f"WHERE trading_day > '{(dt.datetime.today() - relativedelta(years=5)).strftime('%Y-%m-%d')}' "
                     f"AND currency_code ='HKD'", conn)
    print(df.shape, df.describe())
    extra = {'con': conn_test, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 1000}
    df.to_sql("data_price", **extra)
global_vals.engine.dispose()
test_engine.dispose()