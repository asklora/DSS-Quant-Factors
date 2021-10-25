import global_vals
import datetime as dt
import pandas as pd
import os

def remove_tables_with_suffix(engine, suffix):
    '''
    Remove all tables in db engine with names ending with the value of `suffix`.

    Usage: python -c "from utils import remove_tables_with_suffix; \\
        from global_vals import engine_ali; \\
        remove_tables_with_suffix(engine_ali, '_matthew');"
    '''
    if suffix:
        with engine.connect() as conn:
            tbls = conn.execute(f"select table_name from information_schema.tables where table_name like '%%{suffix}';")
            for (tbl,) in tbls:
                cmd = f'drop table {tbl};'
                conn.execute(f'drop table {tbl};')
                print(cmd)
    else:
        print("Do not remove any tables because the suffix is an empty string.")

def record_table_update_time(tb_name, conn):
    ''' record last update time in table '''
    update_time = dt.datetime.now()
    try:
        delete_query_history = f"DELETE FROM {global_vals.update_time_table} WHERE index='{tb_name}'"
        conn.execute(delete_query_history)
    except Exception as e:
        print(e)
    extra = {'con': conn, 'index': False, 'if_exists': 'append'}
    df = pd.DataFrame({'update_time': {tb_name: update_time}}).reset_index()
    df.to_sql(global_vals.update_time_table, **extra)

if __name__ == "__main__":
    pass
    # read_from_firebase()