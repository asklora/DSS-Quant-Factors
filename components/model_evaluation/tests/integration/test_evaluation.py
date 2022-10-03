from components.model_evaluation.src.evaluate_factor_premium import evalFactor
from components.model_evaluation.src.evaluate_top_selection import evalTop


processes = 10
eval_name_sql = 'w8_20221002025855'


def test_eval():
    eval_df = evalFactor(name_sql=eval_name_sql,
                         processes=processes).write_db()

    assert len(eval_df) > 0


def test_eval_top():
    score_df = evalTop(name_sql=eval_name_sql,
                       processes=processes).write_top_select_eval()

    assert len(score_df) > 0


def test_eval_select():
    select_df = evalTop(name_sql=eval_name_sql,
                        processes=processes).write_latest_select()

    assert len(select_df) > 0
