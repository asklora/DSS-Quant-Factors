# before test on dev DB needs to migrate macro / formula table to DB

import pandas as pd
from sqlalchemy import Table, create_engine, MetaData
from sqlalchemy.dialects.postgresql import TEXT, JSON
import global_vars


def migrate_schema():
    """ create table used in this repo in Dev DB"""

    engine_dev = create_engine(global_vars.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
    engine_prod = create_engine(global_vars.db_url_alibaba_prod, max_overflow=-1, isolation_level="AUTOCOMMIT")

    metadata = MetaData()
    for k, v in global_vars.__dict__.items():
        if "table" in k:
            print(f'----> create table: {v}')
            table = Table(v, metadata, autoload=True, autoload_with=engine_prod)
    metadata.create_all(engine_dev)

    engine_dev.dispose()
    engine_prod.dispose()


def migrate_tables():
    """ migrate tables used in script other than calculation_ratio.py to Dev DB """

    engine_dev = create_engine(global_vars.db_url_alibaba, max_overflow=-1, isolation_level="AUTOCOMMIT")
    engine_prod = create_engine(global_vars.db_url_alibaba_prod, max_overflow=-1, isolation_level="AUTOCOMMIT")

    with engine_prod.connect() as conn_prod, engine_dev.connect() as conn_dev:
        for t in ['data_vix', 'data_macro', 'factor_formula_ratios_prod', 'factor_formula_y_type']:
            print(f'----> migrate data: {t}')
            df = pd.read_sql(f'SELECT * FROM {t}', conn_prod)

            extra = {'con': conn_dev, 'index': False, 'if_exists': "append", 'method': 'multi', 'chunksize': 20000}
            if t == 'factor_formula_y_type':
                extra["dtype"] = {"y_type": TEXT, "factor_list": JSON}
            df.to_sql(t, **extra)

    engine_dev.dispose()
    engine_prod.dispose()


def get_table_dtypes(table_name):
    """ create table used in this repo in Dev DB"""

    engine_prod = create_engine(global_vars.db_url_alibaba_prod, max_overflow=-1, isolation_level="AUTOCOMMIT")

    dtypes = {}
    metadata = MetaData()
    table = Table(table_name, metadata, autoload=True, autoload_with=engine_prod)
    for c in table.c:
        dtypes[c.name] = c.type
    print(dtypes)

    engine_prod.dispose()


if __name__ == '__main__':
    # migrate_schema()
    # migrate_tables()
    get_table_dtypes('factor_result_rank_backtest_eval')