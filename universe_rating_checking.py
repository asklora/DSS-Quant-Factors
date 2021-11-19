import pandas as pd
import datetime as dt
import global_vars
from utils_sql import sql_read_query, sql_read_table

rating = sql_read_query("SELECT ticker, trading_day, ai_score, wts_rating FROM universe_rating_history", global_vals.db_url_aws_read)
sector = sql_read_query("SELECT ticker, icb_code as sector FROM universe WHERE currency_code = 'USD' and icb_code<>'NA'", global_vals.db_url_aws_read)

rating = rating.merge(sector, on=["ticker"])
rating["sector"] = rating["sector"].astype(str).str[:4].astype(int)
rating = rating.sort_values(by=['ai_score'], ascending=False).groupby(["trading_day"]).head(20)
rating = rating.sort_values(by=["trading_day", "ai_score"], ascending=False)

icb_name = sql_read_query("SELECT DISTINCT code_4 as sector, name_4 as sector_name FROM icb_code_explanation", global_vals.db_url_alibaba)
icb_name["sector"] = icb_name["sector"].astype(int)
rating = rating.merge(icb_name, on=["sector"])

from collections import Counter
for p in rating["trading_day"].unique():
    df = rating.loc[rating["trading_day"]==p]
    print(p, list(Counter(df["sector_name"]))[:3])