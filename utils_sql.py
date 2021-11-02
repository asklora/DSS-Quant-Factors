from pangres import upsert
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy import create_engine
import pandas as pd
import datetime as dt

import global_vals
from utils_report_to_slack import to_slack

def trucncate_table_in_database(table, db_url=global_vals.db_url_alibaba):
    ''' truncate table in DB (for tables only kept the most recent model records) -> but need to keep table structure'''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(f"TRUNCATE TABLE {table}")
        print(f"TRUNCATE TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN TRUNCATE DB === Error : {e}")

def drop_table_in_database(table, db_url=global_vals.db_url_alibaba):
    ''' drop table in DB (for tables only kept the most recent model records) '''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        print(f"DROP TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN DROP DB === Error : {e}")

def upsert_data_to_database(data, table, primary_key, db_url=global_vals.db_url_alibaba, how="update",
                            drop_primary_key=False):
    ''' upsert Table to DB '''

    try:
        print(f"=== Upsert Data to Database on Table [{table}] ===")
        print(f"=== URL: {db_url} ===")

        if len(primary_key) > 1:    # for tables using more than 1 columns as primary key (replace primary with a created "uid")
            data = uid_maker(data, primary_key)
            primary_key = "uid"
            if drop_primary_key:    # drop original columns (>1) for primary keys
                data = data.drop(columns=primary_key)

        # df = data.duplicated(subset=[primary_key], keep=False)
        # df = data.loc[df]

        if data.duplicated(subset=[primary_key], keep=False).sum()>0:
            to_slack("clair").message_to_slack(f"Exception: duplicated on primary key: [{primary_key}]")

        data = data.drop_duplicates(subset=[primary_key], keep="first", inplace=False)
        data = data.dropna(subset=[primary_key])
        data = data.set_index(primary_key)
        data_type = {primary_key: TEXT}

        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        if how in ["replace", "append"]:
            extra = {'con': engine.connect(), 'index': False, 'if_exists': how, 'method': 'multi', 'chunksize':20000}
            data.to_sql(table, **extra)
        else:
            upsert(engine=engine,
                   df=data,
                   table_name=table,
                   if_row_exists=how,
                   chunksize=20000,
                   dtype=data_type)
            print(f"DATA [{how}] TO {table}")
        engine.dispose()
        to_slack("clair").message_to_slack(f"===  FINISH [{how}] DB [{table}] ===")
        record_table_update_time(table)
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN UPSERT DB === Error : {e}")

def sql_read_query(query, db_url=global_vals.db_url_alibaba):
    ''' Read specific query from SQL '''

    print(f'      ------------------------> Download Table with query: [{query}]')
    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, chunksize=10000)
        df = pd.concat(df, axis=0, ignore_index=True)
    engine.dispose()
    return df

def sql_read_table(table, db_url=global_vals.db_url_alibaba):
    ''' Read entire table from SQL '''

    print(f'      ------------------------> Download Entire Table from [{table}]')
    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, chunksize=10000)
        df = pd.concat(df, axis=0, ignore_index=True)
    engine.dispose()
    return df

def record_table_update_time(table):
    ''' record last update time in table '''
    update_time = dt.datetime.now()
    df = pd.DataFrame({'table_name': table, 'last_update': update_time, 'finish': True}, index=[0]).set_index("table_name")

    engine = create_engine(global_vals.db_url_alibaba_prod, max_overflow=-1, isolation_level="AUTOCOMMIT")
    upsert(engine=engine,
           df=df,
           table_name=global_vals.update_time_table,
           if_row_exists="update",
           chunksize=20000)
    engine.dispose()

def uid_maker(df, primary_key):
    ''' create uid columns for table when multiple columns used as primary_key '''
    df["uid"] = df[primary_key].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
    for s in [" ", ",", ":", ".", "-", "'","_"]:
        df["uid"] = df["uid"].str.replace(s, "")
    return df

if __name__=="__main__":
    df = sql_read_table("iso_currency_code")
    uid_maker(df, ["nation_code","nation_name"])
    # data = data.loc[data['nation_code']=='372']
    # data['currency_code'] = "test"
    # upsert_data_to_database(data, "iso_currency_code1", "nation_code", how="update")
