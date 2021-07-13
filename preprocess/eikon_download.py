import eikon as ek
import pandas as pd
import global_vals
import datetime as dt
import numpy as np

if __name__ == '__main__':
    with global_vals.engine.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker FROM {global_vals.stock_data_table}", conn)
        tickers = list(set(universe['ticker'].to_list()))
    global_vals.engine.dispose()
    # tickers = pd.read_csv('non_ticker.csv')['0'].to_list()

    print(tickers)
    # exit(1)

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    tickers=['9999.HK']
    df_list = []
    step = 39
    params = {'SDate': '2000-01-01', 'EDate': '2021-07-01', 'Frq': 'FQ'}

    params = {'SDate': '2019-12-31', 'Scales':'6'}

    fields=[
            'TR.PCRetEarnTot',
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
            'TR.F.PrefStockRedeemConvert'
            'TR.F.IncTaxDef',
    ]
    #65391137000
    # display(df)

    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        try:
            df, err = ek.get_data(ticker, fields=fields) #, parameters=params)
        except Exception as e:
            print(ticker, e)
            continue
        df_list.append(df)
        print(pd.concat(df_list, axis=0))
        print(err)
    ddf = pd.concat(df_list, axis=0).transpose()
    print(ddf)
    ddf.to_csv('stock_price.csv')

