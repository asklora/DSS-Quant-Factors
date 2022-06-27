def test_download_sql_query():
    from components.model_evaluation.src.calculation_rank import cleanSubpillar

    df = calcRank(name_sql="w8_20220428152336_debug", eval_factor=True)._download_prediction_from_db()

    assert len(df) > 0

