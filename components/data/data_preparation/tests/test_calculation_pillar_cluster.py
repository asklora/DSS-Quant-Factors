import pandas as pd
from utils import dateNow
import datetime as dt


def test_calcPillarCluster_download_pivot_ratio():
    from components.data.data_preparation.src.calculation_pillar_cluster import calcPillarCluster

    df = calcPillarCluster(4, 4, ["USD"])._download_pivot_ratio()

    df = df.reset_index()
    assert len(df) > 0
    assert df["trading_day"].max() == pd.date_range(end=dateNow(), periods=1, freq='W-Sun')[0]


def test_calcPillarCluster():
    from components.data.data_preparation.src.calculation_pillar_cluster import calcPillarCluster

    df = calcPillarCluster(4, 4, ["USD"]).write_all()

    assert len(df) > 0
    assert df["testing_period"].max() == pd.date_range(end=dateNow(), periods=5, freq='W-Sun')[0]
