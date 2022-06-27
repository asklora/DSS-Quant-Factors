import pandas as pd
from utils import dateNow


def test_calcPillarCluster_testing_period_list():
    from components.data_preparation.src.calculation_pillar_cluster import calcPillarCluster

    cls = calcPillarCluster(4, 4, ["USD"])

    from utils import backdate_by_day
    end_date = pd.date_range(end=pd.date_range(end=backdate_by_day(1), periods=1, freq='W-MON')[0], periods=2, freq='4W-SUN')[0]
    assert max(cls.period_list) == end_date


def test_calcPillarCluster_download_pivot_ratio():
    from components.data_preparation.src.calculation_pillar_cluster import calcPillarCluster

    df = calcPillarCluster(4, 4, ["USD"])._download_pivot_ratio()

    df = df.reset_index()
    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


def test_calcPillarCluster():
    from components.data_preparation.src.calculation_pillar_cluster import calcPillarCluster

    df = calcPillarCluster(4, 4, ["USD"]).write_all()

    assert len(df) > 0
    assert df["testing_period"].max() == pd.date_range(end=dateNow(), periods=5, freq='W-Sun')[0]
