import pandas as pd
import numpy as np


def test_get_industry_name_map():
    from components.model_evaluation.src.evaluate_top_selection import get_industry_name_map
    dict = get_industry_name_map()

    assert dict


def test_EvalTop__calc_agg_returns():
    from components.model_evaluation.src.evaluate_top_selection import EvalTop

    df = pd.DataFrame(
        [[['assets_1yr'], ['assets_1yr'], -0.01, -0.02],
         [np.nan, ['cash_ratio'], -0.01, 0.0],
         [['tax_less_pension_to_accu_depre'], np.nan, -0.0, -0.03],
         [['ca_turnover'], [], -0.0, 0.03]],
        columns=["max_factor", "min_factor", "max_ret", "min_ret"]
    )

    cls = EvalTop(name_sql='w8_20220629115419', processes=1)
    df = df.assign(**{x: 1 for x in cls.eval_config_define_columns + cls.eval_config_opt_columns})

    df = cls._EvalTop__calc_agg_returns(df)

    assert len(df) == 1


def test_EvalTop_write_latest_select():
    from components.model_evaluation.src.evaluate_top_selection import EvalTop
    select_df = EvalTop(name_sql="w8_20220712105506", processes=1).write_latest_select()

    assert len(select_df) > 0


def test_EvalTop_write_top_select_eval():
    from components.model_evaluation.src.evaluate_top_selection import EvalTop
    select_df = EvalTop(name_sql="w8_20220630155536", processes=1).write_top_select_eval()

    assert len(select_df) > 0
