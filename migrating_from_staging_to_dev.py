import os
from utils import read_query, models, upsert_data_to_database
from sqlalchemy import select, func
if __name__=='__main__':
    os.environ['DEBUG']='False'
    # breakpoint()
    # migrate()
    # query = select(models.Universe)
    query = f"SELECT * FROM factor.factor_formula_config_train;"
    df = read_query(query=query)
    # df.to_pickle('factor_processed_ratio.pkl')
    # breakpoint()
    os.environ['DEBUG']='True'

    table='factor.factor_formula_config_train'
    upsert_data_to_database(df,table=table)

    # os.environ['DEBUG']='True'
    # query =f"SELECT testing_period,max_ret,min_ret, currency_code, pillar FROM factor.factor_result_rank_backtest_eval WHERE factor_reverse='0' AND average_days = '-7' AND eval_q = '0.33' AND eval_removed_subpillar='true' AND factor_pca='0.4' AND y_qcut = '0' and valid_pct = '0.2' and valid_method ='2014' AND down_mkt_pct='0.5' "
    # df = read_query(query)
    # breakpoint()


# import psycopg2
# from guppy import hpy
# h = hpy()

# conn = psycopg2.connect(
#     host='192.168.1.151',
#     database='quant_dev',
#     user='quant_factor',
#     password='quant_factor'
# )

# print("Connection made")
# print(h.heap())

# cur = conn.cursor()

# query = "SELECT * FROM factor.factor_processed_ratio WHERE trading_day BETWEEN '2007-01-01' AND '2022-01-01';"

# print("Executing query")
# cur.execute(query)
# print(h.heap())
# print("Fetching all")
# cur.fetchall()
# print(h.heap())
