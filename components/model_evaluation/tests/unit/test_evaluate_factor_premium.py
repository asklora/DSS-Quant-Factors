import datetime as dt
import numpy as np
import pandas as pd


# def test_cleanSubpillar_download_subpillars():
#     from components.model_evaluation.src.evaluate_factor_premium import cleanSubpillar
#
#     df = cleanSubpillar(weeks_to_expire=8, start_date=dt.datetime(1998, 4, 24))._download_subpillars()
#     assert len(df) > 0
#
#
# def test_cleanSubpillar_get_subpillar():
#     from components.model_evaluation.src.evaluate_factor_premium import cleanSubpillar
#
#     df = cleanSubpillar(weeks_to_expire=8, start_date=dt.datetime(1998, 4, 24)).get_subpillar()
#     df = df.sort_values(by=["testing_period", "currency_code", "subpillar"], ascending=False)
#     assert len(df) > 0
#     assert set(df.columns.to_list()) == {"testing_period", "currency_code", "factor_name", "subpillar"}
#
#     df_non_dup = df.loc[df.duplicated(["testing_period", "currency_code", "factor_name"], keep=False)]
#     assert len(df_non_dup) == 0
#
#
# def test_cleanPrediction_get_prediction():
#     from components.model_evaluation.src.evaluate_factor_premium import cleanPrediction
#
#     df = cleanPrediction(name_sql='w4_20220627171203').get_prediction()
#     assert len(df) > 0
#
#
# def test_EvalFactor__filter_sample():
#     from components.model_evaluation.src.evaluate_factor_premium import EvalFactor, cleanPrediction, cleanSubpillar
#     from components.model_evaluation.src.load_eval_configs import load_eval_config
#
#     all_groups = load_eval_config(weeks_to_expire=8)
#
#     name_sql = 'w4_20220627171813'
#     pred_df = cleanPrediction(name_sql=name_sql).get_prediction()
#
#     eval_cls = EvalFactor(name_sql='w4_20220627171813', processes=1, all_groups=all_groups)
#     df = eval_cls._EvalFactor__filter_sample(df=pred_df, currency_code="EUR", pillar="value")
#
#     assert len(df) > 0
#     assert len(df) < len(pred_df)
#
#
# def test_EvalFactor__map_remove_subpillar():
#     from components.model_evaluation.src.evaluate_factor_premium import EvalFactor, cleanPrediction, cleanSubpillar
#     from components.model_evaluation.src.load_eval_configs import load_eval_config
#
#     all_groups = load_eval_config(weeks_to_expire=8)
#
#     name_sql = 'w4_20220627171813'
#     weeks_to_expire = 4
#     pred_df = cleanPrediction(name_sql=name_sql).get_prediction()
#     subpillar_df = cleanSubpillar(start_date=pred_df["testing_period"].min(), weeks_to_expire=weeks_to_expire).get_subpillar()
#
#     eval_cls = EvalFactor(name_sql='w4_20220627171813', processes=1, all_groups=all_groups)
#     sample_df = eval_cls._EvalFactor__filter_sample(df=pred_df, currency_code="EUR", pillar="value")
#     df = eval_cls._EvalFactor__map_remove_subpillar(df=sample_df, subpillar_df=subpillar_df, pillar="value")
#
#     assert len(df) > 0
#     assert "subpillar" in df.columns.to_list()
#     assert len(df) < len(sample_df)
#
#
# def sample_g():
#     return pd.DataFrame(
#         [['assets_1yr', 'assets_1yr', -0.01, -0.02],
#          ['cash_ratio', 'cash_ratio', -0.01, 0.0],
#          ['tax_less_pension_to_accu_depre', 'tax_less_pension_to_accu_depre', -0.0, -0.03],
#          ['ca_turnover', 'subpillar_2', -0.0, 0.03],
#          ['fwd_ey', 'subpillar_0', -0.0, 0.02],
#          ['dividend_1yr', 'dividend_1yr', -0.0, -0.03],
#          ['inv_turnover', 'inv_turnover', 0.0, 0.01],
#          ['ebtda_1yr', 'ebtda_1yr', 0.0, -0.04],
#          ['capex_to_dda', 'capex_to_dda', 0.0, -0.03],
#          ['fwd_roic', 'subpillar_3', 0.0, 0.0],
#          ['debt_to_asset', 'debt_to_asset', 0.0, 0.01],
#          ['fa_turnover', 'subpillar_1', 0.01, 0.02],
#          ['epsq_1q', 'epsq_1q', 0.01, -0.01],
#          ['interest_to_earnings', 'interest_to_earnings', 0.01, 0.01],
#          ['div_payout', 'div_payout', 0.02, 0.0]],
#         columns=["factor_name", "subpillar", "pred", "actual"]
#     )
#
#
# def test_groupSummaryStats_get_stats_multioutput():
#     from components.model_evaluation.src.evaluate_factor_premium import groupSummaryStats
#
#     g = sample_g()
#     results = groupSummaryStats(eval_q=0.33, eval_removed_subpillar=True).get_stats(g)
#
#     assert len(results) == 13
#
#
# def test_groupSummaryStats_get_stats_singleoutput():
#     from components.model_evaluation.src.evaluate_factor_premium import groupSummaryStats
#
#     g = sample_g().iloc[[0], :]
#     results = groupSummaryStats(eval_q=0.33, eval_removed_subpillar=True).get_stats(g)
#
#     assert len(results) == 6
#
#
# def test_groupSummaryStats_get_stats_multioutput_no_actual():
#     from components.model_evaluation.src.evaluate_factor_premium import groupSummaryStats
#
#     g = sample_g()
#     g["actual"] = np.nan
#
#     results = groupSummaryStats(eval_q=0.33, eval_removed_subpillar=True).get_stats(g)
#
#     assert len(results) == 6
#
#
# def test_EvalFactor_rank_pillar_cluster():
#     from components.model_evaluation.src.evaluate_factor_premium import EvalFactor, cleanPrediction, cleanSubpillar
#     from components.model_evaluation.src.load_eval_configs import load_eval_config
#
#     all_groups = load_eval_config(weeks_to_expire=8)
#
#     name_sql = 'w8_20220629115419'
#     weeks_to_expire = 4
#     pred_df = cleanPrediction(name_sql=name_sql).get_prediction()
#     subpillar_df = cleanSubpillar(start_date=pred_df["testing_period"].min(), weeks_to_expire=weeks_to_expire).get_subpillar()
#
#     eval_cls = EvalFactor(name_sql='w8_20220629115419', processes=1, all_groups=all_groups)
#
#     df1 = eval_cls._rank(next(i[0] for i in all_groups if i[0]["currency_code"] == "HKD"), pred_df=pred_df, subpillar_df=subpillar_df)
#     assert len(df1) > 0
#     assert {"eval_q", "eval_remove_subpillar"}.issubset(set(df1.columns.to_list()))


def test_EvalFactor():
    from components.model_evaluation.src.evaluate_factor_premium import EvalFactor
    eval_df = EvalFactor(name_sql="w8_20220629115419", processes=2).write_db()

    assert len(eval_df) > 0
