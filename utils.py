import global_vals
import datetime as dt
import pandas as pd
import os
from pangres import upsert

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
    df = pd.read_sql(f'SELECT * FROM {global_vals.update_time_table}', conn)
    df.loc[df['index']==tb_name, 'update_time'] = update_time
    extra = {'con': conn, 'index': False, 'if_exists': 'append'}
    df.to_sql(global_vals.update_time_table, **extra)

def read_from_firebase():

    # Import database module.
    import firebase_admin
    from firebase_admin import credentials, firestore

    # Get a database reference to our posts
    if not firebase_admin._apps:
        cred = credentials.Certificate(global_vals.firebase_url)
        default_app = firebase_admin.initialize_app(cred)

    db = firestore.client()
    doc_ref = db.collection(u"universe").get()

    object_list = []
    for data in doc_ref:
        format_data = {}
        data = data.to_dict()
        format_data['ticker'] = data.get('ticker')
        format_data['negative_factor'] = data.get('rating', {}).get('negative_factor', 0)
        format_data['positive_factor'] = data.get('rating', {}).get('positive_factor', 0)
        format_data['ai_score'] = data.get('rating', {}).get('ai_score', 0)
        object_list.append(format_data)

    result = pd.DataFrame(object_list)
    return result

if __name__ == "__main__":
    read_from_firebase()