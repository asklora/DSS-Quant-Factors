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