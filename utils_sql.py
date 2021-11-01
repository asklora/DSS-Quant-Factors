from pangres import upsert
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy import create_engine
import pandas as pd

import global_vals
from utils_report_to_slack import to_slack

def trucncate_table_in_database(table, db_url=global_vals.db_url_alibaba):
    ''' truncate table in DB (for tables only kept the most recent model records) -> but need to keep table structure'''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with global_vals.engine_ali.connect() as conn:
            conn.execute(f"TRUNCATE TABLE IF EXISTS {table}")
        print(f"TRUNCATE TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN TRUNCATE DB === Error : {e}")

def drop_table_in_database(table, db_url=global_vals.db_url_alibaba):
    ''' drop table in DB (for tables only kept the most recent model records) '''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with global_vals.engine_ali.connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        print(f"DROP TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN DROP DB === Error : {e}")

def upsert_data_to_database(data, table, primary_key, db_url=global_vals.db_url_alibaba, how="update"):
    ''' upsert Table to DB '''

    try:
        print(f"=== Upsert Data to Database on Table [{table}] ===")
        print(f"=== URL: {db_url} ===")
        if data.duplicated(subset=[primary_key], keep=False):
            raise Exception(f"Exception: duplicated on primary key: [{primary_key}]")
        data = data.drop_duplicates(subset=[primary_key], keep="first", inplace=False)
        data = data.dropna(subset=[primary_key])
        data = data.set_index(primary_key)
        data_type = {primary_key: TEXT}

        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        upsert(engine=engine,
               df=data,
               table_name=table,
               if_row_exists=how,
               chunksize=10000,
               dtype=data_type)
        print(f"DATA UPSERT TO {table}")
        engine.dispose()
        to_slack("clair").message_to_slack(f"===  FINISH [{how}] DB [{table}] ===")
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN UPSERT DB === Error : {e}")

def sql_read_query(query, db_url=global_vals.db_url_alibaba):
    ''' Read specific query from SQL '''

    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    engine.dispose()
    return df

def sql_read_table(table, db_url=global_vals.db_url_alibaba):
    ''' Read entire table from SQL '''

    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    engine.dispose()
    return df

if __name__=="__main__":
    drop_table_in_database("iso_currency_code1")
    # data = data.loc[data['nation_code']=='372']
    # data['currency_code'] = "test"
    # upsert_data_to_database(data, "iso_currency_code1", "nation_code", how="update")
