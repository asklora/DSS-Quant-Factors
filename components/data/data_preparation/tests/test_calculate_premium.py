def test_download_pivot_ratios():
    from components.data.data_preparation.src.calculation_premium import calcPremium
    df = calcPremium(weeks_to_expire=8, average_days=[-7], weeks_to_offset=4, processes=1, all_train_currencys=["USD"]
                     )._download_pivot_ratios()

    assert len(df) > 0


def test_calcPremium():
    from components.data.data_preparation.src.calculation_premium import calcPremium
    df = calcPremium(weeks_to_expire=8, average_days=[-7], weeks_to_offset=4, processes=1, all_train_currencys=["USD"]).get_all()

    assert len(df) > 0
