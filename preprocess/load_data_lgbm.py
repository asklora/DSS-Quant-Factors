import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

import global_vals

def download_clean_macros():
    ''' download macros data from DB and preprocess: convert some to yoy format '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download macro data from {global_vals.macro_data_table}')

    with global_vals.engine.connect() as conn:
        macros = pd.read_sql(f'SELECT * FROM {global_vals.macro_data_table} WHERE period_end IS NOT NULL', conn)
    global_vals.engine.dispose()

    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')

    yoy_col = macros.select_dtypes('float').columns[macros.select_dtypes('float').mean(axis=0) > 100]  # convert YoY
    num_col = macros.select_dtypes('float').columns.to_list()  # all numeric columns

    macros[yoy_col] = (macros[yoy_col] / macros[yoy_col].shift(12)).sub(1)  # convert yoy_col to YoY
    macros["period_end"] = macros['trading_day'] + MonthEnd(0)              # convert timestamp to monthend

    return macros.drop(['trading_day','data'], axis=1)

def download_index_return():
    ''' download index return data from DB and preprocess: convert to YoY and pivot table '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download index return data from {global_vals.processed_ratio_table}')

    with global_vals.engine_ali.connect() as conn:
        index_ret = pd.read_sql(f"SELECT ticker, period_end, stock_return_r1_0, stock_return_r6_2, stock_return_r12_7 "
                                f"FROM {global_vals.processed_ratio_table} WHERE ticker like '.%%'", conn)
    global_vals.engine.dispose()

    # index_to_curr = {'.TWII':'TWD',
    #                 '.N225':'JPY',
    #                 '.KS200':'KRW',
    #                 '.FTSE':'GBP',
    #                 '.HSI':'HKD',
    #                 '.STI':'SGD',
    #                 '.SXXGR':'EUR',
    #                 '.CSI300':'CNY',
    #                 '.SPX':'USD'}
    # index_ret['ticker'] = index_ret['ticker'].replace(index_to_curr)

    stock_return_col = ['stock_return_r1_0', 'stock_return_r6_2', 'stock_return_r12_7']
    index_ret[stock_return_col] = index_ret[stock_return_col] + 1
    index_ret['return'] = np.prod(index_ret[stock_return_col].values, axis=1) - 1
    index_ret = pd.pivot_table(index_ret, columns=['ticker'], index=['period_end'], values='return').reset_index(drop=False)

    return index_ret.filter(['period_end','.SPX','.HSI','.CSI300'])      # Currency only include USD/HKD/CNY indices

def combine_data():
    ''' combine factor premiums with ratios '''

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM {global_vals.factor_premium_table} WHERE \"group\" IS NOT NULL', conn)
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
    global_vals.engine.dispose()

    factors = formula.loc[formula['factors'], 'name'].to_list()         # remove factors no longer used
    df = df.filter(['group','period_end'] + factors)
    df['period_end'] = pd.to_datetime(df['period_end'])                 # convert to datetime

    df = df.loc[df['period_end'] < dt.datetime.today() + MonthEnd(-2)]  # remove

    # Add Macroeconomic variables - from Datastream
    macros = download_clean_macros()
    df = df.merge(macros, on=['period_end'], how='left')

    # Add index return variables
    index_ret = download_index_return()
    df = df.merge(index_ret, on=['period_end'], how='left')

    return df.sort_values(by=['group', 'period_end']), factors

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self):
        ''' split train and testing set
                    -> return dictionary contain (x, y, y without qcut) & cut_bins'''

        # define self objects
        self.sample_set = {}
        self.group = pd.DataFrame()
        self.main, self.factor_list = combine_data()    # combine all data

    def split_group(self, group_name=None):
        ''' split main sample sets in to industry_parition or country_partition '''

        if group_name == 'industry':
            self.group = self.main.loc[self.main['group'].str[-1]=='0']          # train on industry partition factors
        elif group_name == 'currency':
            self.group = self.main.loc[self.main['group'].str[-1]!='0']          # train on currency partition factors
        else:
            raise ValueError("Invalid group_name method. Expecting 'industry' or 'currency' got ", group_name)

    def split_train_test(self, testing_period, ar_period, ma_period, y_type):
        ''' split training / testing set based on testing period '''

        # Calculate the time_series history for predicted Y
        for i in range(1, ar_period+1):
            ar_col = [f"ar_{x}_{i}m" for x in y_type]
            self.group[ar_col] = self.group.groupby(['group'])[y_type].shift(i)

        # Calculate the moving average for predicted Y
        ma_col = [f"ma_{x}_{ma_period}m" for x in y_type]
        self.group.loc[:, ma_col] = self.group.groupby(['group'])[y_type].transform(lambda x: x.rolling(ma_period, min_periods=6).mean())

        # Calculate the predicted Y
        y_col = ["y_" + x for x in y_type]
        self.group.loc[:, y_col] = self.group.groupby(['group'])[y_type].shift(-1).values

        # split training/testing sets based on testing_period
        start = testing_period - relativedelta(years=10)    # train df = 40 quarters
        self.train = self.group.loc[(start <= self.group['period_end']) &
                                     (self.group['period_end'] < testing_period)].reset_index(drop=True)
        self.train = self.train.dropna(subset=y_col)      # remove training sample with NaN Y

        self.test = self.group.loc[self.group['period_end'] == testing_period].reset_index(drop=True)

        def divide_set(df):
            ''' split x, y from main '''
            return df.iloc[:, 2:-1].values, df[y_col].values     # Assuming using all factors

        self.sample_set['train_x'], self.sample_set['train_y'] = divide_set(self.train)
        self.sample_set['test_x'], self.sample_set['test_y'] = divide_set(self.test)

    def standardize_x(self):
        ''' standardize x with train_x fit '''

        scaler = StandardScaler().fit(self.sample_set['train_x'])
        self.sample_set['train_x'] = scaler.transform(self.sample_set['train_x'])
        self.sample_set['test_x'] = scaler.transform(self.sample_set['test_x'])

    def y_qcut(self, qcut_q=3):
        ''' convert qcut bins to median of each group '''
        self.sample_set['train_y_final'] = np.zeros(self.sample_set['train_y'].shape)
        self.sample_set['train_y_final'][:] = np.nan
        self.sample_set['test_y_final'] = np.zeros(self.sample_set['test_y'].shape)
        self.sample_set['test_y_final'][:] = np.nan

        for i in range(self.sample_set['train_y'].shape[1]):
            # cut original series into bins
            self.sample_set['train_y_final'][:,i], cut_bins = pd.qcut(self.sample_set['train_y'][:,i], q=qcut_q,
                                                                      retbins=True, labels=False, duplicates='drop')
            cut_bins[0], cut_bins[-1] = [-np.inf, np.inf]
            self.sample_set['test_y_final'][:,i] = pd.cut(self.sample_set['test_y'][:,i], bins=cut_bins, labels=False)

    def split_valid(self, n_splits):
        ''' split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set '''

        gkf = GroupShuffleSplit(n_splits=n_splits).split( self.sample_set['train_x'],
                                                          self.sample_set['train_y_final'],
                                                          groups=self.train['group'])

        return gkf

    def split_all(self, testing_period, y_type, qcut_q=3, n_splits=5, ar_period=12, ma_period=12):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, ar_period, ma_period, y_type)   # split x, y for test / train samples
        self.standardize_x()                                          # standardize x array
        self.y_qcut(qcut_q)                                           # qcut and median convert y array
        gkf = self.split_valid(n_splits)                              # split for cross validation in groups

        return self.sample_set, gkf

if __name__ == '__main__':
    # download_index_return()
    testing_period = dt.datetime(2019,7,31)
    y_type = ['earnings_yield','market_cap_usd']
    group_code = 'industry'

    data = load_data()
    data.split_group(group_code)
    sample_set, cv = data.split_all(testing_period, y_type=y_type)

    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)