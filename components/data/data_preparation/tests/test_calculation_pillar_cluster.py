def test_calcPillarCluster_download_pivot_ratio():
    from components.data.data_preparation.src.calculation_pillar_cluster import calcPillarCluster
    import datetime as dt

    df = calcPillarCluster([dt.datetime(2022, 4, 10)], 4)._download_pivot_ratio()

    assert len(df) > 0



def test_calcPillarCluster():
    from components.data.data_preparation.src.calculation_pillar_cluster import calcPillarCluster
    import datetime as dt

    results = calcPillarCluster([dt.datetime(2022, 4, 10)], 4).write_all()

    assert len(results) > 0
