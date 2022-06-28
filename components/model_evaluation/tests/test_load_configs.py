from typing import List


def test_get_latest_name_sql():
    from components.model_evaluation.src.load_eval_configs import load_latest_name_sql
    name = load_latest_name_sql(8)

    assert type(name) != type(None)


def test_load_eval_config():
    from components.model_evaluation.src.load_eval_configs import load_eval_config
    lst = load_eval_config(4)

    assert len(lst) > 0
    assert isinstance(lst, list)
    assert isinstance(lst[0], tuple)
    assert isinstance(lst[0][0], dict)