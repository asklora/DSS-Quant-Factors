import eikon as ek
import global_vars
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import os

def download_from_eikon_others():
    ''' Monthly Update: download report_date from eikon '''

    os.chdir('C:\\Users\\clair\\PycharmProjects\\factors\\descriptive_factor\\fund name')

    # lst = []
    # for f in os.listdir():
    #     df = pd.read_excel(f, sheet_name=0)
    #     lst.append(df)
    # df = pd.concat(lst, axis=0)
    # df['RIC'].to_csv('Fund_RIC.csv', index=False)

    df = pd.read_csv('Fund_RIC.csv').dropna(how='any')
    print(df.dtypes)

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')
    tickers = df['RIC'].to_list()

    end = dt.datetime.today()
    start = dt.datetime.today()
    params = {'SDate': start.strftime('%Y-%m-%d'), 'EDate': end.strftime('%Y-%m-%d'), 'Frq': 'D'}      # params for fundemantals
    fields_step = {"TR.FundHoldingRIC": 500} # , "TR.FundTopTenHoldings.allocationname": 1000, "TRFundTotalNetAssets.": 500

    for fields, step in fields_step.items():
        for i in np.arange(0, len(tickers),step):
            ticker = tickers[i:(i + step)]
            print(i, ticker)
            try:
                df, err = ek.get_data(ticker, fields=[fields], parameters=params)
            except Exception as e:
                print(e)
                continue
            print(err)
            df.columns = ['ticker', 'fx_rate']
            df = df.dropna(how='any')

            # write to DB
            with global_vals.engine_ali.connect() as conn:
                extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
                df.to_sql(global_vals.eikon_other_table+f"_{fields.split('.')[1]}", **extra)
            global_vals.engine_ali.dispose()

    # # drop duplicates
    # with global_vals.engine_ali.connect() as conn:
    #     all = pd.read_sql(f'SELECT * FROM {global_vals.eikon_other_table}_fx', conn)
    #     extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
    #     all_unique = all.drop_duplicates(keep='last')
    #     all_unique.to_sql(global_vals.eikon_other_table + '_fx', **extra)
    # global_vals.engine_ali.dispose()

def reverse_fmt():
    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql("SELECT * FROM {}".format(global_vals.eikon_other_table + '_fx'), conn)
    global_vals.engine_ali.dispose()

    df.loc[df['ticker'].isin(['AUD']), 'fx_rate'] = 1/df.loc[df['ticker'].isin(['AUD']), 'fx_rate']

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        df.to_sql(global_vals.eikon_other_table + '_fx', **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    download_from_eikon_others()
    # reverse_fmt()