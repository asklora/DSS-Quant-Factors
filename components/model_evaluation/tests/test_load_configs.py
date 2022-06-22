def test_get_latest_name_sql():
    from components.model_evaluation.src.load_configs import load_latest_name_sql
    name = load_latest_name_sql(8)

    assert type(name) != type(None)


def test_load_eval_config():
    from components.model_evaluation.src.load_configs import load_eval_config
    df = load_eval_config(4)

    assert len(df) > 0