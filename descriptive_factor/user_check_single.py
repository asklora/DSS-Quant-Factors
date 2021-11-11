import global_vals
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from utils_sql import sql_read_query, sql_read_table

def check_user(user_id):
    ''' check all transaction of single user '''

    # get all position_uid from orders_position
    op_query = f"SELECT opp.created, op.bot_id, op.position_uid, op.ticker, opp.current_pnl_amt, opp.order_summary, " \
               f"opp.status, opp.share_num, opp.current_investment_amount, opp.current_bot_cash_balance, " \
               f"opp.last_spot_price, opp.last_live_price, op.event " \
               f"FROM orders_position op "
    op_query += f"INNER JOIN orders_position_performance opp ON op.position_uid=opp.position_uid "
    op_query += f"WHERE user_id='{user_id}' AND opp.created>='2021-11-01' "
    op_query += f"ORDER BY op.ticker, op.position_uid, opp.created"

    op = sql_read_query(op_query, global_vals.db_url_aws_read)
    print(op["current_pnl_amt"].sum())

    # evaluate top earner by (ticker / bot) for this user
    op_last = op.groupby("position_uid").last()
    op_ticker_sum = op_last.groupby("ticker").sum()
    op_bot_sum = op_last.groupby("bot_id").sum()

    # evaluate whether buy/sell amount match with profit
    # qty_cols = ["created", "position_uid","ticker","share_num", "order_summary","event", "status"]
    op["created"] = pd.to_datetime(op["created"]).dt.tz_localize(None)
    # op.loc[op['status']=="Populate", "created"] = dt.datetime(2000,1,1,0,0,0)
    op = op.sort_values(by=["position_uid", "created"])
    op["trans_num"] = op["share_num"] - op.groupby("position_uid")["share_num"].shift(1)
    op["trans_num"] = op["trans_num"].fillna(op["share_num"])
    op["trans_profit"] = op["trans_num"]*(op["last_spot_price"]-op["last_live_price"])
    print(op["trans_profit"].sum())

    # evaluate price used for amount calculation is correct
    op["trading_day"] = op["created"].dt.date
    price_query = "SELECT trading_day, ticker, high, low FROM master_ohlcvtr WHERE trading_day>='2021-10-01'"
    price = sql_read_query(price_query, global_vals.db_url_aws_read)
    op = op.merge(price, on=["trading_day","ticker"])
    op["correct"] = ((op["last_live_price"]<=op["high"])&(op["last_live_price"]>=op["low"]))|(op["low"].isnull())
    print(op)

def check_ticker(ticker):
    rating_query = f"SELECT * FROM universe_rating_history WHERE ticker='{ticker}' ORDER BY trading_day DESC"
    rating = sql_read_query(rating_query, global_vals.db_url_aws_read)

    des_query = f"SELECT * FROM universe WHERE ticker='{ticker}'"
    universe = sql_read_query(des_query, global_vals.db_url_aws_read)

    return rating, universe

if __name__=="__main__":
    # check_ticker("1211.HK")
    check_user(user_id=1423)
