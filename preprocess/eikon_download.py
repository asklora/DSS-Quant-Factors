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

    print(tickers)
    # exit(1)

    ek.set_app_key('5c452d92214347ec8bd6270cab734e58ec70af2c')

    # tickers=['AAPL.O']
    df_list = []
    step = 39
    params = {'SDate': '2000-01-01', 'EDate': '2021-07-01', 'Frq': 'M'}

    for i in np.arange(0, len(tickers),step):
        ticker = tickers[i:(i + step)]
        print(i, ticker)
        try:
            df, err = ek.get_data(ticker, fields=['TR.MarketCapLocalCurn(Curn=Native,Scale=6)','TR.MarketCapLocalCurn.date'], parameters=params)
        except:
            print('ERROR: ', ticker)
            continue
        if err is None:
            df_list.append(df)
            print(pd.concat(df_list, axis=0))
        print(err)
    ddf = pd.concat(df_list, axis=0)
    ddf.to_csv('mktcap.csv')


