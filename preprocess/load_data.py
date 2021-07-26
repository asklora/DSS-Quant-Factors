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

    with global_vals.engine_ali.connect() as conn:
        macros = pd.read_sql(f'SELECT * FROM {global_vals.macro_data_table} WHERE period_end IS NOT NULL', conn)
    global_vals.engine_ali.dispose()

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
    global_vals.engine_ali.dispose()

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

def download_org_ratios(method='median', change=True):
    ''' download the aggregated value of all original ratios by each group '''

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {global_vals.processed_group_ratio_table} WHERE method = '{method}'", conn)
    global_vals.engine_ali.dispose()
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y-%m-%d')
    field_col = df.columns.to_list()[2:-1]

    if change:  # calculate the change of original ratio from T-1 -> T0
        df[field_col] = df[field_col]/df.sort_values(['period_end']).groupby(['group'])[field_col].shift(1)-1
    df.columns = df.columns.to_list()[:2] + ['org_'+x for x in field_col] + [df.columns.to_list()[-1]]

    return df.iloc[:,:-1]

def combine_data(use_biweekly_stock, stock_last_week_avg):
    ''' combine factor premiums with ratios '''

    if use_biweekly_stock and stock_last_week_avg:
        raise ValueError("Expecting 'use_biweekly_stock' or 'stock_last_week_avg' is TRUE. Got both is TRUE")

    # Read sql from different tables
    factor_table_name = global_vals.factor_premium_table
    if use_biweekly_stock:
        factor_table_name+='_biweeky'
    elif stock_last_week_avg:
        factor_table_name+='_weekavg'

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM {factor_table_name} WHERE \"group\" IS NOT NULL', conn)
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
    global_vals.engine_ali.dispose()

    factors = formula.sort_values(by=['rank']).loc[formula['factors'], 'name'].to_list()         # remove factors no longer used
    df = df.filter(['group','period_end'] + factors)
    df['period_end'] = pd.to_datetime(df['period_end'])                 # convert to datetime

    df = df.loc[df['period_end'] < dt.datetime.today() + MonthEnd(-2)]  # remove

    # Add Macroeconomic variables - from Datastream
    macros = download_clean_macros()
    df = df.merge(macros, on=['period_end'], how='left')

    # Add index return variables
    index_ret = download_index_return()
    df = df.merge(index_ret, on=['period_end'], how='left')

    # Add original ratios variables
    org_df = download_org_ratios()
    df = df.merge(org_df, on=['group', 'period_end'], how='left')

    return df.sort_values(by=['group', 'period_end']), factors

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self, use_biweekly_stock=False, stock_last_week_avg=False):
        ''' combine all possible data to be used '''

        # define self objects
        self.sample_set = {}
        self.group = pd.DataFrame()
        self.main, self.factor_list = combine_data(use_biweekly_stock, stock_last_week_avg)    # combine all data

        self.all_y_col = ["y_" + x for x in self.factor_list]    # calculate y for all factors
        self.main[self.all_y_col] = self.main.groupby(['group'])[self.factor_list].shift(-1)

    def split_group(self, group_name=None):
        ''' split main sample sets in to industry_parition or country_partition '''

        curr_list = ['TWD','JPY','KRW','GBP','HKD','SGD','EUR','CNY','USD']

        if group_name == 'currency':
            self.group = self.main.loc[self.main['group'].isin(curr_list)]          # train on industry partition factors
        else:
            self.group = self.main.loc[~self.main['group'].isin(curr_list)]          # train on currency partition factors

    def split_train_test(self, testing_period, y_type, ar_period, ma_period):
        ''' split training / testing set based on testing period '''

        # Calculate the time_series history for predicted Y
        for i in range(1, ar_period + 1):
            ar_col = [f"ar_{x}_{i}m" for x in y_type]
            self.main[ar_col] = self.main.groupby(['group'])[y_type].shift(i)

        # Calculate the moving average for predicted Y
        ma_col = [f"ma_{x}_{ma_period}m" for x in y_type]
        self.main.loc[:, ma_col] = self.main.groupby(['group'])[y_type].transform(
            lambda x: x.rolling(ma_period, min_periods=6).mean())

        y_col = ["y_" + x for x in y_type]

        # split training/testing sets based on testing_period
        start = testing_period - relativedelta(years=10)    # train df = 40 quarters
        self.train = self.group.loc[(start <= self.group['period_end']) &
                                     (self.group['period_end'] < testing_period)]
        self.train = self.train.dropna(subset=y_col).reset_index(drop=True)      # remove training sample with NaN Y

        self.test = self.group.loc[self.group['period_end'] == testing_period].reset_index(drop=True)

        self.y_qcut_all()   # qcut/cut for all factors to be predicted at the same time

        def divide_set(df):
            ''' split x, y from main '''

            def find_col(df, k):
                l = df.columns.to_list()
                return [x for x in l if k in x]

            x_col = set(df.columns.to_list()) - set(find_col(df, "y_") + find_col(df, "org_") + ['period_end','group'])
            x_col = list(x_col) + ["org_"+x for x in y_type]
            y_col_cut = [x+'_cut' for x in y_col]
            return df[x_col].values, df[y_col].values, df[y_col_cut].values, x_col     # Assuming using all factors

        self.sample_set['train_x'], self.sample_set['train_y'], self.sample_set['train_y_final'],_ = divide_set(self.train)
        self.sample_set['test_x'], self.sample_set['test_y'], self.sample_set['test_y_final'], self.x_col = divide_set(self.test)

    def standardize_x(self):
        ''' standardize x with train_x fit '''

        scaler = StandardScaler().fit(self.sample_set['train_x'])
        self.sample_set['train_x'] = scaler.transform(self.sample_set['train_x'])
        self.sample_set['test_x'] = scaler.transform(self.sample_set['test_x'])

    def y_qcut_all(self, qcut_q=3):
        ''' convert continuous Y to discrete (0, 1, 2) for all factors during the training / testing period '''

        cut_col = [x+"_cut" for x in self.all_y_col]
        arr = self.train[self.all_y_col].values.flatten()       # Flatten all training factors to qcut all together
        # arr[(arr>np.quantile(np.nan_to_num(arr), 0.99))|(arr<np.quantile(np.nan_to_num(arr), 0.01))] = np.nan

        # cut original series into bins
        arr, self.cut_bins = pd.qcut(arr, q=qcut_q, retbins=True, labels=False)
        # arr, cut_bins = pd.cut(arr, bins=3, retbins=True, labels=False)
        self.train[cut_col] = np.reshape(arr, (len(self.train), len(self.all_y_col)), order='C')
        print(self.cut_bins)

        # def count_i(df, l):
        #     s = df.isnull().sum(axis=0)
        #     n = df.count()
        #     df = df.replace(l, np.nan)
        #     return (df.isnull().sum(axis=0) - s)/n
        # ddf = pd.DataFrame(self.sample_set['train_y_final'], columns=self.factor_list)
        # c1 = count_i(ddf, [1,2,3])
        # print(c1)

        self.cut_bins[0], self.cut_bins[-1] = [-np.inf, np.inf]
        arr_test = self.test[self.all_y_col].values.flatten()       # Flatten all testing factors to qcut all together
        arr_test = pd.cut(arr_test, bins=self.cut_bins, labels=False)
        self.test[cut_col] = np.reshape(arr_test, (len(self.test), len(self.all_y_col)), order='C')

    def split_valid(self, testing_period, n_splits, valid_method):
        ''' split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set '''

        if valid_method == "cv":       # split validation set by cross-validation 5 split
            gkf = GroupShuffleSplit(n_splits=n_splits).split( self.sample_set['train_x'],
                                                              self.sample_set['train_y_final'],
                                                              groups=self.train['group'])
        elif valid_method == "chron":       # split validation set by chronological order
            valid_period = testing_period - relativedelta(years=2)   # using last 2 year samples as valid set
            test_index = self.train.loc[self.train['period_end'] >= valid_period].index.to_list()
            train_index = self.train.loc[self.train['period_end'] < valid_period].index.to_list()
            gkf = [(train_index, test_index)]
        else:
            raise ValueError("Invalid valid_method. Expecting 'cv' or 'chron' got ", valid_method)

        return gkf

    def split_all(self, testing_period, y_type, qcut_q=3, n_splits=5, ar_period=12, ma_period=12, valid_method='cv'):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, y_type, ar_period, ma_period)   # split x, y for test / train samples
        self.standardize_x()                                          # standardize x array
        gkf = self.split_valid(testing_period, n_splits, valid_method)                              # split for cross validation in groups

        return self.sample_set, gkf

if __name__ == '__main__':
    # download_org_ratios('mean')
    # download_index_return()
    testing_period = dt.datetime(2019,7,31)
    y_type = ['earnings_yield','market_cap_usd']
    group_code = 'industry'

    data = load_data(stock_last_week_avg=True)
    data.split_group(group_code)
    sample_set, cv = data.split_all(testing_period, y_type=y_type)

    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)