import eikon as ek
import global_vals
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

def download_from_eikon_others():
    ''' Monthly Update: download report_date from eikon '''

    # with global_vals.engine.connect() as conn:
    #     universe = pd.read_sql(f"SELECT ticker FROM {global_vals.dl_value_universe_table}", conn)
    #     tickers = list(universe['ticker'].unique())
    # global_vals.engine.dispose()
#
    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')
    tickers = ['CNY=', "HKD=", "GBP=", "EUR=", "KRW="]

    step = 1
    end = dt.datetime.today()
    start = dt.date(1998,1,1)
    params = {'SDate': start.strftime('%Y-%m-%d'), 'EDate': end.strftime('%Y-%m-%d'), 'Frq': 'D'}      # params for fundemantals
    fields = ['TR.MIDPRICE', 'TR.MIDPRICE.date']

    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        try:
            df, err = ek.get_data(ticker, fields=fields, parameters=params)
        except Exception as e:
            print(ticker, e)
            continue
        df.columns = ['ticker', 'fx_rate', 'period_end']
        df = df.dropna(how='any')
        df['ticker'] = df['ticker'].str[:-1]
        df['period_end'] = pd.to_datetime(df['period_end'])

        # write to DB
        with global_vals.engine_ali.connect() as conn:
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
            df.to_sql(global_vals.eikon_other_table+'_fx', **extra)
        global_vals.engine_ali.dispose()

def reverse_fmt():
    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql("SELECT * FROM {}".format(global_vals.eikon_other_table + '_fx'), conn)
    global_vals.engine_ali.dispose()

    df.loc[df['ticker'].isin(['GBP','EUR']), 'fx_rate'] = 1/df.loc[df['ticker'].isin(['GBP','EUR']), 'fx_rate']

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        df.to_sql(global_vals.eikon_other_table + '_fx', **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    # download_from_eikon_others()
    reverse_fmt()reverse_fmt