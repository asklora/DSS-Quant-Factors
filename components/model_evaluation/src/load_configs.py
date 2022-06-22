import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
from itertools import product
from sqlalchemy import select, and_, func, text

from utils import (
    to_slack,
    report_to_slack,
    read_query,
    read_table,
    upsert_data_to_database,
    models,
    sys_logger,
    err2slack,
    dateNow
)

logger = sys_logger(__name__, "DEBUG")

config_eval_table = models.FactorFormulaEvalConfig.__table__.schema + '.' + models.FactorFormulaEvalConfig.__table__.name


def load_eval_config(weeks_to_expire):
    eval_configs = read_query(f"SELECT * FROM {config_eval_table} WHERE is_active AND weeks_to_expire = {weeks_to_expire}")
    eval_configs = eval_configs.drop(columns=["is_active"]).to_dict("records")
    assert len(eval_configs) > 0    # else no training will be done
    logger.info(f"=== evaluation iteration: n={len(eval_configs)} ===")

    return [tuple([e]) for e in eval_configs]


def load_latest_name_sql(weeks_to_expire):

    query = select(models.FactorResultScore.name_sql).where(
        models.FactorResultScore.name_sql.like(f'w{weeks_to_expire}_%%')
    ).order_by(models.FactorResultScore.uid.desc()).limit(1)

    df = read_query(query)

    return df.iloc[0, 0]