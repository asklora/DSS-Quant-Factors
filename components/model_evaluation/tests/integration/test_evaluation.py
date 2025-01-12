from components.model_evaluation.src.evaluate_factor_premium import EvalFactor
from components.model_evaluation.src.evaluate_top_selection import EvalTop


processes = 1
eval_name_sql = 'w8_20221018013845'


def test_eval():
    eval_df = EvalFactor(name_sql=eval_name_sql,
                         processes=processes).write_db()

    assert len(eval_df) > 0


def test_eval_top():
    score_df = EvalTop(name_sql=eval_name_sql,
                       processes=processes).write_top_select_eval()

    assert len(score_df) > 0


def test_eval_select():
    select_df = EvalTop(name_sql=eval_name_sql,
                        processes=processes).write_latest_select()

    assert len(select_df) > 0
