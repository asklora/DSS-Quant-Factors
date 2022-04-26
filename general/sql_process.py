from multiprocessing import cpu_count
from pangres import upsert
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy import create_engine
import pandas as pd
import datetime as dt
from retry import retry

import global_vars
from global_vars import *
from general.send_slack import to_slack

logger = logger(__name__, LOGGER_LEVEL)


# ============================================ EXECUTE QUERY ================================================

def trucncate_table_in_database(table, db_url=db_url_write):
    """ truncate table in DB (for tables only kept the most recent model records) -> but need to keep table structure """
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(f"TRUNCATE TABLE {table}")
        logger.info(f"TRUNCATE TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN TRUNCATE DB [{table}] === Error : {e}")


def drop_table_in_database(table, db_url=db_url_write):
    ''' drop table in DB (for tables only kept the most recent model records) '''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            conn.execute(f"DROP TABLE {table}")
        logger.info(f"DROP TABLE: [{table}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN DROP DB [{table}] === Error : {e}")


def delete_data_on_database(table, db_url=db_url_write, query=None):
    ''' delete data from table in databased '''
    try:
        engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT")
        if type(query) == type(None):
            query = "True"
        with engine.connect() as conn:
            conn.execute(f"DELETE FROM {table} WHERE {query}")
        logger.info(f"DELETE TABLE: [{table}] WHERE [{query}]")
        engine.dispose()
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN DELETE DB [{table}] === Error : {e}")
        return False
    return True


# ============================================== WRITE TABLE ==================================================

@retry(tries=3, delay=1)
def upsert_data_to_database(data, table, schema='factor', primary_key=["uid"], db_url=db_url_write, how="update",
                            verbose=1, dtype=None, chunksize=20000, mp=False):
    """ upsert Table to DB

    Parameters
    ----------
    data (DataFrame):
        data to write to DB Table
    table (Str, Optional):
        DB table name
    schema (Str, Optional):
        DB schema name
    db_url :
        write to which DB (default=Alibaba Dev)
    primary_key (List[Str], Optional):
        Primary key of the data (compulsory when how=update/ignore)
    how (Str, Optional):
        how to write to DB (default=update)
    verbose (Float, Optional):
        if True, report write to DB to slack (default=1, i.e. report to DB)
    """

    try:
        logger.info(f"=== [{how}] Data (n={len(data)}) to Database on Table [{table}] ===")
        logger.info(f"=== URL: {db_url} ===")

        engine = create_engine(db_url, pool_size=cpu_count() if mp else 1, max_overflow=-1, isolation_level="AUTOCOMMIT")
        if how in ["replace", "append"]:
            with engine.connect() as conn:
                extra = {'con': conn, 'index': False, 'if_exists': how, 'method': 'multi', 'chunksize': chunksize,
                         'dtype': dtype, "schema": schema}
                data.to_sql(table, **extra)
            engine.dispose()
        else:
            if data.duplicated(subset=primary_key, keep=False).sum() > 0:
                to_slack("clair").message_to_slack(f"Exception: duplicated on primary key: {primary_key}")

            data = data.drop_duplicates(subset=primary_key, keep="first", inplace=False)
            data = data.dropna(subset=primary_key)
            data = data.set_index(primary_key)

            if type(dtype) != type(None):
                upsert_params = dict(
                    df=data,
                    table_name=table,
                    if_row_exists=how,
                    chunksize=chunksize,
                    add_new_columns=True,
                    dtype=dtype,
                    schema=schema
                )
            else:
                upsert_params = dict(
                    df=data,
                    table_name=table,
                    if_row_exists=how,
                    chunksize=chunksize,
                    add_new_columns=True,
                    schema=schema
                )
            upsert(engine, **upsert_params)
            logger.debug(f"DATA [{how}] TO {table}")
        engine.dispose()
        if verbose >= 0:
            to_slack("clair").message_to_slack(f"===  FINISH [{how}] DB [{table}] ===")
        record_table_update_time(table)
    except Exception as e:
        to_slack("clair").message_to_slack(f"===  ERROR IN [{how}] DB [{table}] === Error : {e.args}")
        return False
    return True


# =============================================== READ TABLE ==================================================

def read_query(query, db_url=db_url_read, mp=False):
    ''' Read specific query from SQL '''

    logger.info(f'Download Table with query: [{query}]')
    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT", pool_size=cpu_count() if mp else 1)
    with engine.connect() as conn:
        try:
            df = pd.read_sql(query, conn, chunksize=20000, )
        except Exception as e:
            # logger.debug(e)
            query = query.replace("FROM ", "FROM factor.")
            df = pd.read_sql(query, conn, chunksize=20000, )
        df = pd.concat(df, axis=0, ignore_index=True)
    engine.dispose()
    return df


def read_table(table, db_url=db_url_read, mp=True):
    ''' Read entire table from SQL '''

    logger.info(f'Download Entire Table from [{table}]')
    engine = create_engine(db_url, max_overflow=-1, isolation_level="AUTOCOMMIT", pool_size=cpu_count() if mp else 1)
    with engine.connect() as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn, chunksize=10000)
        except Exception as e:
            # logger.debug(e)
            df = pd.read_sql(f"SELECT * FROM factor.{table}", conn, chunksize=10000)
        df = pd.concat(df, axis=0, ignore_index=True)
    engine.dispose()
    return df


def record_table_update_time(table):
    ''' record last update time in table '''
    update_time = dt.datetime.now()
    df = pd.DataFrame({'tbl_name': table, 'last_update': update_time, 'finish': True}, index=[0]).set_index("tbl_name")
    df.index.name = "tbl_name"
    data_type = {"tbl_name": TEXT}

    engine = create_engine(db_url_write, max_overflow=-1, isolation_level="AUTOCOMMIT")
    upsert(engine,
           df=df,
           table_name=update_time_table,
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


def migrate_local_save_to_prod():
    """ during factor training, we save to local in case error stop the process for recovery """

    for t, how in {global_vars.result_score_table: "update",
                   global_vars.result_pred_table: "append",
                   global_vars.feature_importance_table: "append"}.items():
        data = read_table('factor.' + t, db_url=global_vars.db_url_local)
        status = upsert_data_to_database(data, t, how=how)
        if status:
            delete_data_on_database('factor.' + t, db_url=global_vars.db_url_local)
        logger.info(f"-----> local DB save migrate to cloud: {t}")
    return True


if __name__ == "__main__":
    t = 'factor_formula_pillar_cluster'
    data = read_table(t)
    upsert_data_to_database(data, t, how='append')