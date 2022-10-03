import datetime as dt
import pandas as pd
import numpy as np
import argparse
import time
from dateutil.relativedelta import relativedelta
from itertools import product
from sqlalchemy import select, and_, func, text
import os

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
from .configs import LOGGER_LEVELS
logger = sys_logger(__name__, LOGGER_LEVELS.LOAD_EVAL_CONFIGS)

config_eval_table = models.FactorFormulaEvalConfig.__table__.schema + '.' + models.FactorFormulaEvalConfig.__table__.name


def load_eval_config(weeks_to_expire: int):
    """
    load all evaluation configuration for given weeks_to_expire for certain table
    """

    conditions = [
        models.FactorFormulaEvalConfig.weeks_to_expire == weeks_to_expire,
    ]
    if not os.getenv("DEBUG").lower == "true":
        conditions.append(models.FactorFormulaEvalConfig.id == 0)

    query = select(models.FactorFormulaEvalConfig).where(and_(*conditions))

    eval_configs = read_query(query)
    eval_configs = eval_configs.drop(columns=["id"]).to_dict("records")

    if len(eval_configs) <= 0:
        raise ValueError(f"No eval config selected from [{models.FactorFormulaEvalConfig.__tablename__}] by {__name__}")

    logger.info(f"=== evaluation iteration: n={len(eval_configs)} ===")

    return [tuple([e]) for e in eval_configs]


def load_latest_name_sql(weeks_to_expire: int):
    """
    get last training iteration name_sql for given weeks_to_expire
    """

    query = select(models.FactorResultScore.name_sql).where(
        models.FactorResultScore.name_sql.like(f'w{weeks_to_expire}_%%')
    ).order_by(models.FactorResultScore.uid.desc()).limit(1)

    df = read_query(query)

    return df.iloc[0, 0]