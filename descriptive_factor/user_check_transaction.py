import global_vals
import pandas as pd
import datetime as dt
import numpy as np
from utils_sql import sql_read_query, sql_read_table

def check_user(user_id):
    ''' check all transaction of single user '''

    # get all position_uid from orders_position
    op_query = f"SELECT op.position_uid, op.ticker, opp.order_summary, opp.status, opp.share_num, opp.current_investment_amount, opp.current_bot_cash_balance FROM orders_position op "
    op_query += f"INNER JOIN orders_position_performance opp ON op.position_uid=opp.position_uid "
    op_query += f"WHERE user_id='{user_id}' AND opp.created<'2021-11-01' "
    op_query += f"ORDER BY op.ticker, op.position_uid, opp.created"

    op = sql_read_query(op_query, global_vals.db_url_aws_read)

    op_last = op.groupby("position_uid").last()
    op_last[""]

    op.loc[op['status']=="Populate", "qty"] = op['share_num']
    op.loc[op['order_summary'].notnull(), "qty"] = op.loc[op['order_summary'].notnull(), "order_summary"].apply(lambda x: x["hedge_shares"])
    # op = op.dropna(subset=["qty"])
    op["qty_cum"] = op.groupby("ticker")["qty"].apply(np.cumsum)
    print(op)

    opp_query = f"SELECT order_summary"

if __name__=="__main__":
    check_user(user_id=1423)
