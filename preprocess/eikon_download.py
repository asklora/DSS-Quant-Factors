import eikon as ek
import pandas as pd
import global_vals
import datetime as dt
import numpy as np

def download_from_eikon():
    ''' download fields from eikon '''

    with global_vals.engine.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.stock_data_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine.dispose()
    # tickers = pd.read_csv('non_ticker.csv')['0'].to_list()

    print(tickers)
    # exit(1)
    tickers=['UPM.HE', 'HON', 'KNEBV.HE', 'MOCORP.HE', 'GSX', '600068.SS']
    tickers = ['UPM.HE']

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    # tickers=['9999.HK','AAPL.O']
    df_list = []
    step = 2
    # params = {'SDate': '2000-01-01', 'EDate': '2021-07-01', 'Frq': 'FQ'}

    params = {'SDate': '2000-12-31', 'EDate': '2021-07-01', 'Period':'LTM', 'Frq': 'FQ', 'Scale':'6'}

    fields=[
            'TR.PCRetEarnTot.periodenddate',
            'TR.PCRetEarnTot',
            'TR.F.AdExpn',
            'TR.F.SGA',
            'TR.F.PPENetTot',
            'TR.F.ComShrTrezTot',
            'TR.F.SaleOfTangIntangFixedAssetsGL',
            'TR.F.SaleOfTangIntangFixedAssetsGL.periodenddate',

            'TR.F.TradeAcctPbleTot',
            'TR.F.IncTax',
            'TR.F.DebtLTTot',
            'TR.F.PrefStockRedeemTot',
            'TR.F.RentalExpn',
            'TR.F.RcvblTot',
            'TR.F.IncTaxDef',
            'TR.F.TotPensExpn',
            'TR.F.PrefStockRedeemConvert'
            'TR.F.IncTaxDef',
    ]
    # display(df)

    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        try:
            df, err = ek.get_data(ticker, fields=fields, parameters=params)
        except Exception as e:
            print(ticker, e)
            continue
        df_list.append(df)
        print(pd.concat(df_list, axis=0))
        print(err)
        # df.to_csv('17strategies_output.csv')
        # exit(1)
    ddf = pd.concat(df_list, axis=0)
    print(ddf)
    ddf.to_csv('17strategies_output3.csv')

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


    with global_vals.engine.connect() as conn:
        tickers = set(pd.read_sql(f'SELECT ticker FROM {global_vals.dl_value_universe_table}', conn)['ticker'].to_list())
    global_vals.engine.dispose()

    print(tickers-set(mktcap['ticker'].to_list()))
    a_list = tickers-set(df['Instrument'].to_list())
    print(len(a_list), a_list)

    df = df.merge(mktcap, left_on=['Instrument','Period End Date'], right_on=['ticker','trading_day'], how='outer')
    df['trading_day'] = df['trading_day'].fillna(df['Period End Date'])

    df = df.sort_values(by=['ticker','trading_day'])

    # df = df.drop(['Instrument','Period End Date'], axis=1)

    exit(1)


if __name__ == '__main__':
    download_from_eikon()
    # combine_download_files()

