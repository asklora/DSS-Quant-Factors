import global_vars
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from utils_sql import sql_read_query, sql_read_table

def check_user(user_id):
    ''' check all transaction of single user '''

    # get all position_uid from orders_position
    op_query = f"SELECT opp.updated, op.bot_id, op.position_uid, op.ticker, opp.current_pnl_amt, opp.order_summary, " \
               f"opp.status, opp.share_num, opp.current_investment_amount, opp.current_bot_cash_balance, op.investment_amount, " \
               f"opp.last_spot_price, opp.last_live_price, op.event, op.exchange_rate " \
               f"FROM orders_position op "
    op_query += f"INNER JOIN orders_position_performance opp ON op.position_uid=opp.position_uid "
    op_query += f"WHERE user_id='{user_id}' AND opp.updated>='2021-11-01' "
    op_query += f"ORDER BY op.ticker, op.position_uid, opp.updated"

    op = sql_read_query(op_query, global_vars.db_url_aws_read)

    print(op['current_investment_amount'].sum())

    op["date"] = op["updated"].dt.date

    # evaluate top earner by (ticker / bot) for this user
    op_last = op.groupby("position_uid").last()
    op_ticker_sum = op_last.groupby(["ticker", "date"]).sum()
    op_bot_sum = op_last.groupby("bot_id").sum()
    op_last["current_pnl_amt"] = op_last["current_pnl_amt"]*op_last["exchange_rate"]
    print('Total Profit in Nov:', op_last["current_pnl_amt"].sum())

    # evaluate whether buy/sell amount match with profit
    # qty_cols = ["created", "position_uid","ticker","share_num", "order_summary","event", "status"]
    op["updated"] = pd.to_datetime(op["updated"]).dt.tz_localize(None)
    # op.loc[op['status']=="Populate", "created"] = dt.datetime(2000,1,1,0,0,0)
    op = op.sort_values(by=["position_uid", "updated"])
    op["trans_num"] = op["share_num"] - op.groupby("position_uid")["share_num"].shift(1)
    op["trans_num"] = op["trans_num"].fillna(op["share_num"])
    op["trans_profit"] = op["trans_num"]*(op["last_spot_price"]-op["last_live_price"])
    op["trans_profit"] = op.groupby(["position_uid"])["trans_profit"].cumsum()
    print(op["trans_profit"].sum())

    # check whether profit from DB = recalculated profit
    opp_last = op.groupby(["position_uid"]).last()[["current_pnl_amt", "trans_profit"]]
    opp_last["diff"] = opp_last["current_pnl_amt"] - opp_last["trans_profit"]
    opp_last = opp_last.reset_index()

    # evaluate price used for amount calculation is correct
    op["trading_day"] = op["updated"].dt.date
    price_query = "SELECT trading_day, ticker, high, low, \"close\" FROM master_ohlcvtr WHERE trading_day>='2021-10-01' ORDER BY trading_day asc"
    price = sql_read_query(price_query, global_vars.db_url_aws_read).dropna(how='any')
    price["close"] = price.groupby(["ticker"])["close"].shift(1)
    op = op.merge(price, on=["trading_day","ticker"])
    op["correct"] = (op["last_live_price"]==op["close"])|(op["low"].isnull())
    # op["correct"] = ((op["last_live_price"]<=op["high"])&(op["last_live_price"]>=op["low"]))|(op["low"].isnull())
    print(op)

def check_ticker(ticker):
    rating_query = f"SELECT * FROM universe_rating_history WHERE ticker='{ticker}' ORDER BY trading_day DESC"
    rating = sql_read_query(rating_query, global_vars.db_url_aws_read)

    des_query = f"SELECT * FROM universe WHERE ticker='{ticker}'"
    universe = sql_read_query(des_query, global_vars.db_url_aws_read)

    return rating, universe

if __name__=="__main__":
    # check_ticker("1211.HK")
    check_user(user_id=1428)

    # df0 = sql_read_query("SELECT updated as sold_time, position_uid, bot_cash_balance * exchange_rate as final FROM orders_position "
    #                      "WHERE user_id = 1423 and updated >= '2021-11-01' AND event is not null", global_vars.db_url_aws_read)
    # df = sql_read_query("SELECT * FROM user_transaction "
    #                     "WHERE balance_uid='4b91b275c08e4e0ba27eb4f1bfb9d882' and updated>='2021-10-29'", global_vars.db_url_aws_read)
    # df.loc[df['side']=='credit', "amount"] = -df["amount"]
    #
    # pos_details = sql_read_query(f"SELECT * FROM orders_position WHERE user_id = 1423 and updated >= '2021-11-01'", global_vars.db_url_aws_read)
    # pos_details["final_pnl_amount"] = pos_details["final_pnl_amount"]*pos_details["exchange_rate"]
    #
    # details = pd.DataFrame(df["transaction_detail"].to_list())
    # df = pd.concat([df, details], axis=1)
    #
    # df = df.dropna(subset=["position"])
    # df = df.loc[df["event"]=="create"]
    # print(df["amount"].sum())
    #
    # df = df.merge(df0, left_on="position", right_on="position_uid", how="inner")
    # df = df[["position_uid", "updated", "sold_time", "amount", "final"]]
    # df["pnl"] = df["amount"] - df["final"]
    #
    # df = df.merge(pos_details, on="position_uid")[["pnl","final_pnl_amount"]]
    # df["sum"] = df["pnl"] + df["final_pnl_amount"]
    #
    # print(df.sum())