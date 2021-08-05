import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from preprocess.ratios_calculations import calc_factor_variables
from preprocess.premium_calculation import calc_premium_all

def update_premium(update_only=False, save=False, use_cached=False):

    calc_factor_variables(price_sample='last_week_avg', fill_method='fill_all', sample_interval='monthly',
                          use_cached=False, save=False, update=False)
    calc_premium_all(stock_last_week_avg=True, use_biweekly_stock=False, update=False)

if __name__ == "__main__":
    update_premium(update_only=False, save=True, use_cached=False)
