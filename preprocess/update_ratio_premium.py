import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from preprocess.ratios_calculations import calc_factor_variables
from preprocess.premium_calculation import calc_premium_all

def update_premium(update_only=False, use_biweekly_stock=True):

    if use_biweekly_stock:
        calc_factor_variables(price_sample='last_day', fill_method='fill_all', sample_interval='biweekly',
                              use_cached=False, save=False, update=update_only)
        calc_premium_all(stock_last_week_avg=False, use_biweekly_stock=True, update=update_only)

if __name__ == "__main__":
    update_premium()
