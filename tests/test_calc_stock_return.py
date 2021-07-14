from preprocess.ratios_calculations import calc_stock_return
import global_vals
import pandas as pd

if __name__ == "__main__":
    with global_vals.engine.connect() as conn:
        market_cap = pd.read_sql(f'SELECT * FROM {global_vals.eikon_mktcap_table}', conn)
    global_vals.engine.dispose()

    print(market_cap.shape)
    market_cap = market_cap.drop_duplicates()
    print(market_cap.shape)
    market_cap = market_cap.drop_duplicates(['ticker','trading_day'])
    print(market_cap.shape)
    exit(0)

    with global_vals.engine.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        market_cap.to_sql(global_vals.eikon_mktcap_table, **extra)
    global_vals.engine.dispose()