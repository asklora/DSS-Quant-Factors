import global_vals
import pandas as pd
from sqlalchemy import Table, MetaData
from sqlalchemy.orm import sessionmaker

def get_table_name_list(engine):
    ''' get full list of tables in certain DB '''

    with engine.connect() as conn:
        df = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';", conn)
    engine.dispose()

    return sorted(df['table_name'])

def ali_migration_to_prod(migrate_tbl_lst):
    ''' migrate list of cron table (i.e. currenctly used in DROID_v2.1) used to Prod DB '''

    print(' === Alibaba Migrate Dev to Prod Start === ')
    metadata = MetaData()

    for t in migrate_tbl_lst:
        try:    # create new tables if not exist
            table = Table(t, metadata, autoload=True, autoload_with=global_vals.engine_ali)
            table.create(bind=global_vals.engine_ali_prod)
            print('---> Create table for: ', t)
        except Exception as e:
            print(t, e)

        with global_vals.engine_ali.connect() as conn_dev, global_vals.engine_ali_prod.connect() as conn_prod:
            df = pd.read_sql(f'SELECT * FROM {t}', conn_dev, chunksize=10000)
            df = pd.concat(df)
            extra = {'con': conn_prod, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize':10000}
            df.to_sql(t, **extra)
        global_vals.engine_ali.dispose()
        global_vals.engine_ali_prod.dispose()
        print('     ---> Finish migrate: ', t)

    prod_tbl_lst = get_table_name_list(global_vals.engine_ali_prod)
    fail_migrate_tbl = set(migrate_tbl_lst) - set(prod_tbl_lst)
    if len(fail_migrate_tbl)>0:
        raise Exception("Error: Following table hasn't been migrated to Alibaba DB Prod: ", list(fail_migrate_tbl))

if __name__=="__main__":
    # lst = get_table_name_list(global_vals.engine_ali)
    # print(lst)

    # 'factor_formula_ratios_prod', 'test_fundamental_score_current_names',  'factor_result_pred_prod',
    # 'iso_currency_code', 'ai_value_lgbm_pred_final_eps', 'ingestion_name',
    # 'factor_result_pred_prod_monthly1', 'factor_result_pred_prod_weekly1','factor_formula_ratios_prod_test'
    # 'ai_value_formula_ratios'
    # 'data_factor_eikon_others_date', 'data_factor_eikon_others_fx'

    migrate_tbl_lst = ['ai_value_lgbm_pred','ai_value_lgbm_pred_final','ai_value_lgbm_pred_final_eps','ai_value_lgbm_score']
    ali_migration_to_prod(migrate_tbl_lst)