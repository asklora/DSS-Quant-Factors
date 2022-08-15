import os
from utils import read_query, models, upsert_data_to_database
from sqlalchemy import select, func
if __name__=='__main__':
    os.environ['DEBUG']='False'
    # breakpoint()
    # migrate()
    # query = select(models.Universe)
    query = f"SELECT * FROM factor.factor_processed_ratio WHERE ticker = 'DRQ';"
    df = read_query(query=query)
    # df.to_pickle('factor_processed_ratio.pkl')
    # breakpoint()
    os.environ['DEBUG']='True'
    table='factor.factor_processed_ratio'
    upsert_data_to_database(df,table=table)

    # os.environ['DEBUG']='FALSE'
    # query =f"SELECT DISTINCT ticker FROM factor.factor_processed_ratio;"
    # df = read_query(query)
    # breakpoint()
    # print(123)
