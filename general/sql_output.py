from pangres import upsert
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy import create_engine
import pandas as pd
import datetime as dt

import global_vars
from general.utils_report_to_slack import to_slack

def trucncate_table_in_database(table, db_url=global_vars.db_url_alibaba):
    ''' truncate table in DB (for tables only kept the most recent model records) -> but need to keep table structure'''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(f"TRUNCATE TABLE {table}")
        print(f"TRUNCATE TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN TRUNCATE DB [{table}] === Error : {e}")

def drop_table_in_database(table, db_url=global_vars.db_url_alibaba):
    ''' drop table in DB (for tables only kept the most recent model records) '''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(f"DROP TABLE {table}")
        print(f"DROP TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN DROP DB [{table}] === Error : {e}")

def upsert_data_to_database(data, table, primary_key=None, db_url=global_vars.db_url_alibaba, how="update",
                            drop_primary_key=False, verbose=1, try_drop_table=False):
    ''' upsert Table to DB '''

    try:
        print(f"=== Upsert Data to Database on Table [{table}] ===")
        print(f"=== URL: {db_url} ===")

        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        if how in ["replace", "append"]:
            with engine.connect() as conn:
                extra = {'con': conn, 'index': False, 'if_exists': how, 'method': 'multi', 'chunksize': 20000}
                data.to_sql(table, **extra)
            engine.dispose()
        else:
            if len(primary_key) > 1:  # for tables using more than 1 columns as primary key (replace primary with a created "uid")
                data = uid_maker(data, primary_key)
                primary_key = "uid"
                if drop_primary_key:  # drop original columns (>1) for primary keys
                    data = data.drop(columns=primary_key)

            # df = data.duplicated(subset=[primary_key], keep=False)
            # df = data.loc[df]

            if data.duplicated(subset=[primary_key], keep=False).sum() > 0:
                to_slack("clair").message_to_slack(f"Exception: duplicated on primary key: [{primary_key}]")

            data = data.drop_duplicates(subset=[primary_key], keep="first", inplace=False)
            data = data.dropna(subset=[primary_key])
            data = data.set_index(primary_key)
            data_type = {primary_key: TEXT}

            upsert(engine=engine,
                   df=data,
                   table_name=table,
                   if_row_exists=how,
                   chunksize=20000,
                   dtype=data_type)
            print(f"DATA [{how}] TO {table}")
        engine.dispose()
        if verbose>=0:
            to_slack("clair").message_to_slack(f"===  FINISH [{how}] DB [{table}] ===")
        record_table_update_time(table)
    except Exception as e:
        if try_drop_table:      # if error from columns doesn't exist -> could try to drop table and create again
            try:
                drop_table_in_database(table, db_url)
                upsert(engine=engine,
                       df=data,
                       table_name=table,
                       if_row_exists=how,
                       chunksize=20000,
                       dtype=data_type)
                print(f"DATA [{how}] TO {table}")
                engine.dispose()
            except:
                to_slack("clair").message_to_slack(f"===  ERROR IN [{how}] DB [{table}] === Error : {e}")
        else:
            to_slack("clair").message_to_slack(f"===  ERROR IN [{how}] DB [{table}] === Error : {e}")

def sql_read_query(query, db_url=global_vars.db_url_alibaba):
    ''' Read specific query from SQL '''

    print(f'      ------------------------> Download Table with query: [{query}]')
    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, chunksize=10000)
        df = pd.concat(df, axis=0, ignore_index=True)
    engine.dispose()
    return df

def sql_read_table(table, db_url=global_vars.db_url_alibaba):
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
    df = pd.DataFrame({'tbl_name': table, 'last_update': update_time, 'finish': True}, index=[0]).set_index("tbl_name")
    df.index.name = "tbl_name"
    data_type = {"tbl_name": TEXT}

    engine = create_engine(global_vars.db_url_alibaba_prod, max_overflow=-1, isolation_level="AUTOCOMMIT")
    upsert(engine=engine,
           df=df,
           table_name=global_vars.update_time_table,
           if_row_exists="update",
           chunksize=20000,
           dtype=data_type)
    engine.dispose()

def uid_maker(df, primary_key):
    ''' create uid columns for table when multiple columns used as primary_key '''
    df["uid"] = ''
    for i in primary_key:
        index = df[i].copy()
        for r in [" ", ",", ":", ".", "-", "'", "_"]:
            index = index.astype(str).str.replace(r, "", regex=False)
        df["uid"] += index
    return df

if __name__=="__main__":

    df = sql_read_table("universe_rating_history", global_vars.db_url_aws_read)[["ticker", "trading_day", "ai_score"]]
    universe = sql_read_query("SELECT ticker, currency_code FROM universe WHERE currency_code in ('HKD','USD')", global_vars.db_url_aws_read)
    df = df.merge(universe, on=["ticker"])
    df = df.sort_values(by=["ai_score"]).groupby(["currency_code", "trading_day"]).tail(10)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df = df.loc[df["trading_day"].isin([dt.datetime(2021,11,15), dt.datetime(2021,11,8), dt.datetime(2021,11,1), dt.datetime(2021,10,25)])]
    df.to_csv("universe_rating_history.csv")
    print(df)

    df = sql_read_table("iso_currency_code")
    uid_maker(df, ["nation_code","nation_name"])
    # data = data.loc[data['nation_code']=='372']
    # data['currency_code'] = "test"
    # upsert_data_to_database(data, "iso_currency_code1", "nation_code", how="update")
