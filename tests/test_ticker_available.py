import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

def ticker_available():
    ticker = pd.read_csv('mktcap.csv').dropna(how='any')
    ticker = ticker.drop_duplicates()

    ticker.to_csv('mktcap.csv', index=False)
    exit(0)

    ticm = ticker['ticker'].to_list()

    with global_vals.engine.connect() as conn:
        ticu = pd.read_sql(f'SELECT ticker FROM {global_vals.dl_value_universe_table}', conn)['ticker'].to_list()
    global_vals.engine.dispose()

    non = set(ticu) - set(ticm)

    pd.DataFrame(non).to_csv('non_ticker.csv', index=False)
    print(non)
    print(len(non))



if __name__=="__main__":
    ticker_available()
