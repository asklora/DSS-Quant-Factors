import global_vars
import pandas as pd
from general.utils_sql import sql_read_query
from preprocess.calculation_ratio import fill_all_day

def top_pick_return():
    query = f"SELECT u.ticker, trading_day, ai_score, currency_code FROM universe_rating_history ur "
    query += "INNER JOIN universe u on u.ticker=ur.ticker "
    query += "WHERE currency_code in ('HKD')"

    rating = sql_read_query(query, global_vars.db_url_aws_read).dropna(how='any')
    rating = rating.sort_values(["trading_day", "currency_code", "ai_score"], ascending=False).groupby(["trading_day", "currency_code"]).head(10).reset_index()
    rating = rating.loc[pd.to_datetime(rating["trading_day"]).dt.weekday==0]
    rating.to_csv("top_pick_HKD_history.csv")
    print(rating)

    start_date = (rating["trading_day"].min()).strftime("%Y-%m-%d")
    price_query = "SELECT ticker, trading_day, \"open\", \"close\" FROM master_ohlctrv "
    price_query +="WHERE trading_day>'start_date' "
    price = sql_read_query(price_query, global_vars.db_url_aws_read)
    price = fill_all_day(price)
    price["open"]




if __name__=="__main__":
    top_pick_return()