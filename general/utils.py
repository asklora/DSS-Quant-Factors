import global_vars
import datetime as dt
import pandas as pd
import os

def remove_tables_with_suffix(engine, suffix):
    '''
    Remove all tables in db engine with names ending with the value of `suffix`.

    Usage: python -c "from utils import remove_tables_with_suffix; \\
        from global_vars import engine_ali; \\
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
        delete_query_history = f"DELETE FROM {global_vars.update_time_table} WHERE index='{tb_name}'"
        conn.execute(delete_query_history)
    except Exception as e:
        print(e)
    extra = {'con': conn, 'index': False, 'if_exists': 'append'}
    df = pd.DataFrame({'update_time': {tb_name: update_time}}).reset_index()
    df.to_sql(global_vars.update_time_table, **extra)

def to_excel(df_dict, file_name='test'):
    '''  write DataFrames to excel

    Parameters
    ----------
    df_dict (Dict/DataFrame):
        if = Dict, need to be a dictionary of {sheet_name: DataFrame (content)};
        if = DataFrame, will use default sheet_name "Sheet1".
    file_name : file_name to save
    '''

    writer = pd.ExcelWriter(f'{file_name}.xlsx')
    if type(df_dict)==type({}):
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        n = len(df_dict)
    else:
        df_dict.to_excel(writer, sheet_name='Sheet1', index=False)
        n = 1
    writer.save()
    print(f"=== Finish write [{n}] sheet(s) -> '{file_name}.xlsx")

if __name__ == "__main__":
    pass
    # read_from_firebase()