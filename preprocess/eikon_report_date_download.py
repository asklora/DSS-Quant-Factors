import eikon as ek
import global_vars
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

def download_from_eikon_others():
    ''' Monthly Update: download report_date from eikon '''

    with global_vals.engine.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.dl_value_universe_table}", conn)
        tickers = list(universe['ticker'].unique())
    global_vals.engine.dispose()

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    step = 30
    end = dt.datetime.today()
    start = end - relativedelta(months=3)
    params = {'SDate': start.strftime('%Y-%m-%d'), 'EDate': end.strftime('%Y-%m-%d'), 'Period':'FQ0', 'Frq': 'FQ', 'Scale':'6'}      # params for fundemantals
    fields = ['TR.EPSActReportDate', 'TR.EPSActReportDate.periodenddate']

    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        try:
            df, err = ek.get_data(ticker, fields=fields, parameters=params)
        except Exception as e:
            print(ticker, e)
            continue
        df.columns = ['ticker', 'report_date', 'period_end']
        df['report_date'] = pd.to_datetime(df['report_date'].str[:10])
        df['period_end'] = pd.to_datetime(df['period_end'])

        # write to DB
        with global_vals.engine_ali.connect() as conn:
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
            df.to_sql(global_vals.eikon_other_table+'_date', **extra)
        global_vals.engine_ali.dispose()

    # drop duplicates
    with global_vals.engine_ali.connect() as conn:
        all = pd.read_sql(f'SELECT * FROM {global_vals.eikon_other_table}_date', conn)
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
        all_unique = all.drop_duplicates(keep='last')
        unique_groups = all_unique.groupby(['period_end', 'ticker']).count()
        if any(unique_groups['report_date'] > 1):
            print(unique_groups.loc[unique_groups['report_date'] > 1])
            raise ValueError('Same period_end with different report_date!')
        all_unique.to_sql(global_vals.eikon_other_table + '_date', **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    download_from_eikon_others()