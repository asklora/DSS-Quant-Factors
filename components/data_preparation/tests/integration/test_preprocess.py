from components.data_preparation.src.calculation_ratio import calc_factor_variables_multi
from components.data_preparation.src.calculation_premium import CalcPremium
from components.data_preparation.src.calculation_pillar_cluster import CalcPillarCluster
from utils import (
    sys_logger,
    read_query,
    models,
    check_memory,
    backdate_by_day,
    str_to_date
)

all_currency_list = ["USD"]
weeks_to_expire = 4
all_average_days = -7
sample_interval = 4


def test_ratio():
    data = calc_factor_variables_multi(tickers=None,
                                       currency_codes=all_currency_list,
                                       tri_return_only=False,
                                       processes=1)

    assert len(data) > 0


def test_premium():
    data = CalcPremium(weeks_to_expire=weeks_to_expire,
                       weeks_to_offset=4,
                       currency_code_list=all_currency_list,
                       processes=1,
                       factor_list=["fwd_ey"]).write_all()

    assert len(data) > 0


def test_subpillar():
    data = CalcPillarCluster(weeks_to_expire=weeks_to_expire,
                             currency_code_list=all_currency_list,
                             sample_interval=sample_interval,
                             processes=1).write_all()

    assert len(data) > 0
