import global_vars
import pandas as pd
from sqlalchemy import Table, MetaData


def get_table_name_list(engine):
    ''' get full list of tables in certain DB '''

    with engine.connect() as conn:
        df = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';", conn)
    engine.dispose()

    return sorted(df['table_name'])

def ali_migration_to_prod(migrate_tbl_lst, from_url=global_vars.engine_ali, to_url=global_vars.engine_ali_prod, tbl_pivot=False
                          , tbl_index=["ticker", "trading_day"]):
    ''' migrate list of cron table (i.e. currenctly used in DROID_v2.1) used to Prod DB '''

    print(' === Alibaba Migrate Dev to Prod Start === ')
    metadata = MetaData()

    from general.utils_sql import uid_maker

    for t in migrate_tbl_lst:
        try:    # create new tables if not exist
            table = Table(t, metadata, autoload=True, autoload_with=from_url)
            table.create(bind=to_url)
            print('---> Create table for: ', t)
        except Exception as e:
            print(t, e)

        with from_url.connect() as conn_dev, to_url.connect() as conn_prod:
            df = pd.read_sql(f"SELECT * FROM {t} WHERE ticker='AAPL.O'", conn_dev, chunksize=10000)
            df = pd.concat(df)

            if tbl_pivot:
                cols = [x for x in df.select_dtypes(float).columns.to_list() if x not in tbl_index+["uid"]]
                # df = df.set_index(tbl_index)[cols].stack()
                df = pd.melt(df, id_vars=tbl_index, value_vars=cols, var_name="field", value_name="value").dropna(how='any')
                df = uid_maker(df, primary_key=tbl_index+["field"])
                extra = {'con': conn_prod, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':10000}
                df.to_sql(t, **extra)
            else:
                extra = {'con': conn_prod, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize':10000}
                df.to_sql(t, **extra)
        from_url.dispose()
        to_url.dispose()
        print('     ---> Finish migrate: ', t)

    prod_tbl_lst = get_table_name_list(to_url)
    fail_migrate_tbl = set(migrate_tbl_lst) - set(prod_tbl_lst)
    if len(fail_migrate_tbl)>0:
        raise Exception("Error: Following table hasn't been migrated to Alibaba DB Prod: ", list(fail_migrate_tbl))

if __name__=="__main__":
    # engine = create_engine(global_vars.db_url_aws_read)
    # lst = get_table_name_list(engine)
    # print(lst)
    # exit(0)

    # 'factor_formula_ratios_prod', 'test_fundamental_score_current_names',  'factor_result_pred_prod',
    # 'iso_currency_code', 'ai_value_lgbm_pred_final_eps', 'ingestion_name',
    # 'factor_result_pred_prod_monthly1', 'factor_result_pred_prod_weekly1','factor_formula_ratios_prod_test'
    # 'ai_value_formula_ratios'
    # 'data_factor_eikon_others_date', 'data_factor_eikon_others_fx'

    migrate_tbl_lst = ['data_worldscope_summary']
    ali_migration_to_prod(migrate_tbl_lst, from_url=global_vars.engine, to_url=global_vars.engine_ali, tbl_pivot=True, tbl_index=["ticker", "period_end"])
