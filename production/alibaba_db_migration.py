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
    ''' migrate list of production table (i.e. currenctly used in DROID_v2.1) used to Prod DB '''

    print(' === Alibaba Migrate Dev to Prod Start === ')
    # metadata = MetaData()

    for t in migrate_tbl_lst:
        # try:    # create new tables if not exist
        #     table = Table(t, metadata, autoload=True, autoload_with=global_vals.engine_ali)
        #     table.create(bind=global_vals.engine_ali_prod)
        #     print('---> Create table for: ', t)
        # except Exception as e:
        #     pass

        with global_vals.engine_ali.connect() as conn_dev, global_vals.engine_ali_prod.connect() as conn_prod:
            extra = {'con': conn_prod, 'index': False, 'if_exists': 'replace', 'method': 'multi'}
            pd.read_sql(f'SELECT * FROM {t}', conn_dev).to_sql(t, **extra)
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

    migrate_tbl_lst = ['factor_result_pred_prod_monthly1', 'factor_result_pred_prod_weekly1','factor_formula_ratios_prod_test']
    ali_migration_to_prod(migrate_tbl_lst)