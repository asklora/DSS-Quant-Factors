import eikon as ek
import pandas as pd
import global_vals
import datetime as dt
import numpy as np
import os.path

def download_from_eikon():
    ''' download fields from eikon '''

    with global_vals.engine_ali.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.stock_data_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine_eli.dispose()

    save_name = 'eikon_others.csv'
    tickers = check_eikon_full_ticker(save_name)

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    df_list = []
    step = 3
    # params = {'SDate': '2000-01-01', 'EDate': '2021-07-01', 'Frq': 'M', 'Scale':'6'}  # params for Market Cap

    params = {'SDate': '2000-01-01', 'EDate': '2021-07-01', 'Period':'LTM', 'Frq': 'FQ', 'Scale':'6'}      # params for fundemantals

    # fields = ['TR.CompanyMarketCap(Curn=USD)', 'TR.CompanyMarketCap.date']

    field_name = [ 'TR.PCRetEarnTot',
                'TR.F.AdExpn',
                'TR.F.SGA',
                'TR.F.PPENetTot',
                'TR.F.ComShrTrezTot',
                'TR.F.SaleOfTangIntangFixedAssetsGL',
                'TR.F.TradeAcctPbleTot',
                'TR.F.IncTax',
                'TR.F.DebtLTTot',
                'TR.F.PrefStockRedeemTot',
                'TR.F.RentalExpn',
                'TR.F.RcvblTot',
                'TR.F.IncTaxDef',
                'TR.F.TotPensExpn',
                'TR.F.PrefStockRedeemConvert',
                'TR.F.IncTaxDef',]

    for f in field_name:
        fields = [f, f+'.periodenddate']
        for i in np.arange(0, len(tickers),step):
            ticker = tickers[i:(i + step)]
            print(i, ticker)
            try:
                df, err = ek.get_data(ticker, fields=fields, parameters=params)
            except Exception as e:
                print(ticker, e)
                continue
            df.columns = ['ticker', 'value', 'date']
            df['fields'] = f
            df_list.append(df)
            print(df)

            # df.to_csv('17strategies_output.csv')
            # exit(1)

    ddf = pd.concat(df_list, axis=0)
    ddf.to_csv('eikon_new_downloads.csv')

    print(ddf)

    if os.path.isfile(save_name):
        pre_df = pd.read_csv(save_name)
        pd.concat([pre_df, ddf], axis=0).to_csv(save_name, index=False)
    else:
        ddf.to_csv(save_name, index=False)

def download_from_eikon_others():
    ''' download fields from eikon '''

    with global_vals.engine.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.stock_data_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine.dispose()

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    step = 40
    params = {'SDate': '2000-01-01', 'EDate': '2021-07-01', 'Period':'LTM', 'Frq': 'FQ', 'Scale':'6'}      # params for fundemantals
    # field_name = ['TR.F.OthLiab']
    # field_name = ['TR.F.DebtLTSTIssuanceRetTotCF']
    field_name = ['TR.EPSActReportDate']

    tickers = check_eikon_full_ticker(csv_name='eikon_new_downloads1.csv')

    while len(tickers) > 0:
        df_list = []
        for f in field_name:
            fields = [f, f+'.periodenddate']
            for i in np.arange(0, len(tickers),step):
                ticker = tickers[i:(i + step)]
                print(i, ticker)
                try:
                    df, err = ek.get_data(ticker, fields=fields, parameters=params)
                except Exception as e:
                    print(ticker, e)
                    continue
                df.columns = ['ticker', 'value', 'period_end']
                df['fields'] = f
                df_list.append(df)
                print(df)

        ddf = pd.concat(df_list, axis=0).dropna(how='any')
        ddf.to_csv('eikon_new_downloads.csv')

        with global_vals.engine.connect() as conn:
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 1000}
            ddf.to_sql(global_vals.eikon_other_table, **extra)
        global_vals.engine.dispose()

        tickers = check_eikon_full_ticker(ddf=ddf)
        step = round(step/2)

def download_from_eikon_report_date():
    ''' download fields from eikon '''

    with global_vals.engine_ali.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.dl_value_universe_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine_ali.dispose()
    # pd.DataFrame(tickers).to_csv('tickers.csv')
    # exit(1)

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    step = 50
    params = {'SDate': '2009-01-01', 'EDate': '2021-07-01', 'Period':'FY0', 'Frq': 'FQ', 'Scale':'6'}      # params for fundemantals
    # field_name = ['TR.F.OthLiab']
    # field_name = ['TR.F.DebtLTSTIssuanceRetTotCF']
    field_name = ['TR.ExpectedReportDate']

    tickers = check_eikon_full_ticker(csv_name='eikon_new_downloads1.csv')

    # while len(tickers) > 0:
    df_list = []
    for f in field_name:
        fields = [f, f + '.periodenddate']
        for i in np.arange(0, len(tickers),step):
            ticker = tickers[i:(i + step)]
            print(i, ticker)
            try:
                df, err = ek.get_data(ticker, fields=fields, parameters=params)
            except Exception as e:
                print(ticker, e)
                continue
            df.columns = ['ticker', 'value', 'period_end']
            df_list.append(df)
            print(df)

    ddf = pd.concat(df_list, axis=0).dropna(how='any')
    ddf.drop_duplicates().to_csv('eikon_new_downloads.csv')

        # with global_vals.engine_ali.connect() as conn:
        #     extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 1000}
        #     ddf.to_sql("data_factor_eikon_report_date", **extra)
        # global_vals.engine_ali.dispose()
        #
        # tickers = check_eikon_full_ticker(ddf=ddf)
        # step = round(step/2)

def clean_db_eikon_others():
    pass

def combine_download_files():
    ''' combine eikon downloads in csv formats & upload to DB '''

    mktcap = pd.read_csv('mktcap.csv')
    mktcap['trading_day'] = pd.to_datetime(mktcap['trading_day'], format='%m/%d/%Y')

    df = pd.read_csv('17strategies_output.csv')
    df['Period End Date'] = pd.to_datetime(df['Period End Date'], format='%m/%d/%Y')

    df1 = pd.read_csv('17strategies_output1.csv')
    df1['Period End Date'] = pd.to_datetime(df1['Period End Date'], format='%Y-%m-%d')

    df2 = pd.read_csv('17strategies_output2.csv')
    df2['Period End Date'] = pd.to_datetime(df2['Period End Date'], format='%Y-%m-%d')

    df = pd.concat([df,df1,df2],axis=0)

    df = df.merge(mktcap, left_on=['Instrument','Period End Date'], right_on=['ticker','trading_day'], how='outer')
    df['trading_day'] = df['trading_day'].fillna(df['Period End Date'])

    df = df.sort_values(by=['ticker', 'trading_day'])

    # df = df.drop(['Instrument','Period End Date'], axis=1)

    exit(1)

def download_from_eikon_vix():
    ''' download fields from eikon '''

    ticker = ['.VIX']
    fields = ['TR.PriceClose', 'TR.PriceCloseDate']
    params = {'SDate': '2009-01-01', 'EDate': '2021-07-01', 'Period':'FY0', 'Frq': 'W', 'Scale':'0'}      # params for fundemantals

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    df, err = ek.get_data(ticker, fields=fields, parameters=params)

    df.to_csv('eikon_new_downloads.csv')
    print(df)

def download_from_eikon_mktcap():
    ''' download fields from eikon '''

    with global_vals.engine_ali.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.dl_value_universe_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine_ali.dispose()

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    params = {'SDate': '1999-07-30', 'EDate': '2009-07-30', 'Frq': 'W', 'Scale':'6'}      # params for fundemantals
    field_name = ['TR.CompanyMarketCap(Curn=USD)','TR.CompanyMarketCap(Curn=Native)','TR.CompanyMarketCap.date']

    tickers = check_eikon_full_ticker(global_vals.eikon_mktcap_table+'_weekly')

    step = 5
    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        try:
            df, err = ek.get_data(ticker, fields=field_name, parameters=params)
            df.columns = ['ticker', 'market_cap_usd', 'market_cap', 'period_end']
            df['period_end'] = pd.to_datetime(df['period_end'].str[:10], format='%Y-%m-%d')
            df = df.dropna(how='any')
            print(df)
        except Exception as e:
            print(ticker, e)
            continue

        with global_vals.engine_ali.connect() as conn:
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi'}
            df.to_sql(global_vals.eikon_mktcap_table+'_weekly', **extra)
        global_vals.engine_ali.dispose()

def check_eikon_full_ticker(table_name, last_day='2021-07-31'):
    ''' check if download csv file has all ticker in universe '''

    print(table_name)

    with global_vals.engine_ali.connect() as conn:
        csv_ticker = set(pd.read_sql(f"SELECT DISTINCT \"Instrument\" as ticker FROM {table_name}", conn)['ticker'].to_list())
        # df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        tickers = set(pd.read_sql(f'SELECT ticker FROM {global_vals.dl_value_universe_table}', conn)['ticker'].to_list())
    global_vals.engine_ali.dispose()

    # df = df.loc[df.isnull().sum(axis=1)!=0]
    # miss_list = set(df['ticker'].unique())

    miss_list = tickers-set(csv_ticker)
    print(len(miss_list), miss_list)
    if len(miss_list)<3:
        exit(1)

    return list(miss_list)

def download_from_eikon_tri():
    ''' download fields from eikon '''

    with global_vals.engine_ali.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.dl_value_universe_table}", conn)
        tickers = sorted(list(set(universe['ticker'].to_list())))
    global_vals.engine_ali.dispose()

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    params = {'SDate': '1999-07-30', 'EDate': '2021-07-30', 'ADJUSTED': 1, 'Frq':'D'}      # params for fundemantals

    field_name = ['TR.CLOSEPRICE','TR.HIGHPRICE','TR.OPENPRICE']#,'TR.LOWPRICE']

    # tickers = check_eikon_full_ticker(global_vals.eikon_price_table+'_weekavg_final')

    step = 10
    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        for f in field_name:
            try:
                fields = [f, f+'.date']
                df, err = ek.get_data(ticker, fields=fields, parameters=params)
                df.columns = ['ticker', 'value', 'trading_day']
                df['field'] = f
                df['trading_day'] = pd.to_datetime(df['trading_day'].str[:10], format='%Y-%m-%d')
                df = df.dropna(how='any')
                print(df.head(10))
            except Exception as e:
                print(ticker, e)
                continue

            with global_vals.engine_ali.connect() as conn:
                extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize':1000}
                df.to_sql(global_vals.eikon_price_table+'_weekavg_daily', **extra)
            global_vals.engine_ali.dispose()

def download_from_eikon_vol():
    ''' download fields from eikon '''

    with global_vals.engine_ali.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.dl_value_universe_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine_ali.dispose()

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    params = {'SDate': '1999-07-30', 'EDate': '2021-07-30', 'Frq':'D'}      # params for fundemantals

    field_name = ['TR.VOLUME', 'TR.CLOSEPRICE', 'TR.HIGHPRICE', 'TR.OPENPRICE', 'TR.LOWPRICE', 'TR.VOLUME.date']
    tickers = check_eikon_full_ticker(global_vals.eikon_price_table+'_weekavg_vol')
    print(len(tickers))

    step = 2
    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        # for f in field_name:
        try:
            # fields = field_name
            df, err = ek.get_data(ticker, fields=field_name, parameters=params)
            # df.columns = ['ticker', 'value', 'period_end']
            # df['field'] = f
            df['Date'] = pd.to_datetime(df['Date'].str[:10], format='%Y-%m-%d')
            df = df.dropna(how='any')
            print(df.head(10))
        except Exception as e:
            print(ticker, e)
            continue

        with global_vals.engine_ali.connect() as conn:
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize':1000}
            df.to_sql(global_vals.eikon_price_table+'_weekavg_vol', **extra)
        global_vals.engine_ali.dispose()

def org_eikon_tri_volume():
    with global_vals.engine_ali.connect() as conn:
        df2 = pd.read_sql(f"SELECT * FROM {global_vals.eikon_price_table}_weekavg_vol WHERE \"Volume\"<>0", conn)
        df1 = pd.read_sql(f"SELECT * FROM {global_vals.eikon_price_table}_weekavg_daily", conn)
    global_vals.engine_ali.dispose()

    df2.columns = ['ticker','value','trading_day']
    df2['field'] = 'volume'

    df = pd.concat([df1, df2], axis=0)
    df = pd.pivot_table(df, index=['ticker','trading_day'], columns=['field'], values='value').reset_index()

    print(df.head(10))

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        df.to_sql(global_vals.eikon_price_table + '_daily_final', **extra)
    global_vals.engine_ali.dispose()

if __name__ == '__main__':

    # download_from_eikon_mktcap()
    # check_eikon_full_ticker(global_vals.eikon_mktcap_table+'_weekly')

    # download_from_eikon_tri()
    # check_eikon_full_ticker(global_vals.eikon_price_table+'_weekavg_final')

    # download_from_eikon_vol()

    # check_eikon_full_ticker(global_vals.eikon_price_table+'_weekavg_daily')
    # check_eikon_full_ticker(global_vals.eikon_price_table+'_weekavg_vol')

    org_eikon_tri_volume()

    # from pandas.tseries.offsets import MonthEnd
    # df = pd.read_csv('eikon_new_downloads.csv')
    # df['period_end'] = pd.to_datetime(df['period_end']) + MonthEnd(0)
    # df.to_csv('eikon_new_downloads.csv', index=False)