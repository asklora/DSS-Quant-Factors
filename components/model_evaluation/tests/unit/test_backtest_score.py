# def test_scaleFundamentalScore__download_fundamentals_score():
#      from components.model_evaluation.src.calculation_backtest_score import scaleFundamentalScore
#      df = scaleFundamentalScore(start_date='2022-01-01')._scaleFundamentalScore__download_fundamentals_score()
#
#      assert len(df) > 0


def test_scaleFundamentalScore():
     from components.model_evaluation.src.calculation_backtest_score import scaleFundamentalScore
     df = scaleFundamentalScore(start_date='2022-01-01').get()

     assert len(df) > 0
