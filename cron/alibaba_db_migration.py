import global_vars
import pandas as pd
from sqlalchemy import create_engine
from general.sql_output import uid_maker
from multiprocessing import cpu_count
from sqlalchemy.types import DATE, BIGINT, TEXT, INTEGER, BOOLEAN
from pangres import upsert
from general.utils_report_to_slack import to_slack

DB_READ = global_vars.db_url_alibaba_prod
DB_WRITE = global_vars.db_url_alibaba

def get_table_name_list(engine):
    ''' get full list of tables in certain DB '''

    with engine.connect() as conn:
        df = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';", conn)
    engine.dispose()

    return sorted(df['table_name'])

def read_query(query):
    engine = create_engine(DB_READ, pool_size=cpu_count(), max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        data = pd.read_sql(query, con=conn)
    engine.dispose()
    data = pd.DataFrame(data)
    print(data)
    return data

def upsert_to_database(data, table, primary_key, how="update", type=TEXT):
    print(data)
    try:
        engine = create_engine(DB_WRITE, pool_size=cpu_count(), max_overflow=-1, isolation_level="AUTOCOMMIT")
        if how in ["replace", "append"]:
            extra = {'con': engine.connect(), 'index': False, 'if_exists': how, 'method': 'multi', 'chunksize': 20000}
            data.to_sql(table, **extra)
        else:
            print(f"=== Upsert Data to Database on Table {table} ===")
            data = data.drop_duplicates(subset=[primary_key], keep="first", inplace=False)
            data = data.dropna(subset=[primary_key])
            data = data.set_index(primary_key)

            upsert(engine=engine,
                df=data,
                table_name=table,
                if_row_exists=how,
                chunksize=20000,
                dtype={primary_key:type},
                add_new_columns=True)
            print(f"DATA UPSERT TO {table}")
            engine.dispose()
    except Exception as e:
        print(f"===  ERROR IN UPSERT DB === Error : {e}")
        to_slack("clair").message_to_slack(f"===  ERROR IN UPSERT DB === Error : {e}")

def data_factor_eikon_price():
    table_name = "data_factor_eikon_price_daily_final"
    query = f"select * from {table_name}"
    data = read_query(query).reset_index().rename(columns={"index": "id"})

    table_name = "data_factor_eikon_price"
    upsert_to_database(data, table_name, "id", how="append", type=INTEGER)

def data_factor_eikon_fx():
    table_name = "data_factor_eikon_others_fx"
    query = f"select * from {table_name}"
    data = read_query(query).reset_index().rename(columns={"index": "id"})

    table_name = "data_factor_eikon_fx"
    upsert_to_database(data, table_name, "id", how="append", type=INTEGER)

def data_factor_eikon_date():
    table_name = "data_factor_eikon_others_date"
    query = f"select * from {table_name}"
    data = read_query(query).reset_index().rename(columns={"index": "id"})

    table_name = "data_factor_eikon_date"
    upsert_to_database(data, table_name, "id", how="append", type=INTEGER)

if __name__=="__main__":

    # data_factor_eikon_price()
    data_factor_eikon_fx()
    data_factor_eikon_date()
