from components.data_preparation.src.calculation_ratio import calc_factor_variables_multi
from components.data_preparation.src.calculation_premium import CalcPremium
from components.data_preparation.src.calculation_pillar_cluster import CalcPillarCluster
from utils import (
    sys_logger,
    read_query,
    models,
    check_memory,
    backdate_by_day,
    str_to_date,
    get_active_universe
)
import pandas as pd

all_currency_list = ["USD"]
tickers = ["AAPL.O", "TSLA.O"]
# tickers = None

weeks_to_expire = 8
all_average_days = -7
sample_interval = 4
start_date = pd.to_datetime('1998-01-01')
factor_list = ["fa_turnover"]
# factor_list = None


def test_ratio():
    data = calc_factor_variables_multi(tickers=tickers,
                                       currency_codes=all_currency_list,
                                       tri_return_only=False,
                                       processes=1,
                                       start_date=start_date,
                                       factor_list=factor_list)

    assert len(data) > 0


def test_premium():
    data = CalcPremium(weeks_to_expire=weeks_to_expire,
                       weeks_to_offset=4,
                       currency_code_list=all_currency_list,
                       processes=1,
                       factor_list=["fa_turnover"]).write_all()

    assert len(data) > 0


def test_subpillar():
    data = CalcPillarCluster(weeks_to_expire=weeks_to_expire,
                             currency_code_list=all_currency_list,
                             sample_interval=sample_interval,
                             processes=1,
                             start_date=start_date).write_all()

    assert len(data) > 0
