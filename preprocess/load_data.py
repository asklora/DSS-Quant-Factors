import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from preprocess.premium_calculation import calc_premium_all
import global_vals

def download_clean_macros(main_df, use_biweekly_stock):
    ''' download macros data from DB and preprocess: convert some to yoy format '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download macro data from {global_vals.macro_data_table}')

    with global_vals.engine_ali.connect() as conn:
        macros = pd.read_sql(f'SELECT * FROM {global_vals.macro_data_table} WHERE period_end IS NOT NULL', conn)
        vix = pd.read_sql(f'SELECT * FROM {global_vals.eikon_vix_table}', conn)
    global_vals.engine_ali.dispose()

    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')

    yoy_col = macros.select_dtypes('float').columns[macros.select_dtypes('float').mean(axis=0) > 100]  # convert YoY
    num_col = macros.select_dtypes('float').columns.to_list()  # all numeric columns

    macros[yoy_col] = (macros[yoy_col] / macros[yoy_col].shift(12)).sub(1)  # convert yoy_col to YoY

    # combine macros & vix data
    macros["period_end"] = pd.to_datetime(macros["trading_day"])

    if use_biweekly_stock:
        df_date_list = main_df['period_end'].drop_duplicates().sort_values()
        macros = macros.merge(pd.DataFrame(df_date_list.values, columns=['period_end']), on=['period_end'],
                              how='outer').sort_values(['period_end'])
        macros = macros.fillna(method='ffill')
    else:
        macros["period_end"] = macros['trading_day'] + MonthEnd(0)
        vix["period_end"] = pd.to_datetime(vix["period_end"])
        macros = macros.merge(vix, on=['period_end'], how='outer')

    return macros.drop(['trading_day','data'], axis=1)

def download_index_return(use_biweekly_stock, stock_last_week_avg):
    ''' download index return data from DB and preprocess: convert to YoY and pivot table '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download index return data from {global_vals.processed_ratio_table}')

    # read stock return from ratio calculation table
    if use_biweekly_stock:
        db_table_name = global_vals.processed_ratio_table + '_biweekly'
    elif stock_last_week_avg:
        db_table_name = global_vals.processed_stock_table
    else:
        db_table_name = global_vals.processed_ratio_table

    with global_vals.engine_ali.connect() as conn:
        index_ret = pd.read_sql(f"SELECT * FROM {db_table_name} WHERE ticker like '.%%'", conn)
    global_vals.engine_ali.dispose()

    # Index using SPX/HSI/CSI300 returns
    # stock_return_col = ['stock_return_r1_0', 'stock_return_r6_2', 'stock_return_r12_7']
    # major_index = ['period_end','.SPX','.HSI','.CSI300']    # try include major market index first
    # index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]
    # index_ret = index_ret.set_index(['period_end', 'ticker']).unstack()
    # index_ret.columns = [f'{x[1]}_{x[0][13:]}' for x in index_ret.columns.to_list()]
    # index_ret = index_ret.reset_index()
    # index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])

    # Index using all index r1_0 & vol_30_90 for 6 market based on num of ticker
    major_index = ['period_end','.SPX','.CSI300','.N225','.KS200','.SXXGR','.TWII']    # try include major market index first
    index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]
    index_ret = index_ret.set_index(['period_end', 'ticker'])[['stock_return_r1_0','vol_30_90']].unstack()
    index_ret.columns = [f'{x[1]}_{x[0][0]}' for x in index_ret.columns.to_list()]
    index_ret = index_ret.reset_index()
    index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])

    return index_ret

def download_org_ratios(use_biweekly_stock, stock_last_week_avg, method='mean', change=True):
    ''' download the aggregated value of all original ratios by each group '''

    db_table = global_vals.processed_group_ratio_table
    if stock_last_week_avg:
        db_table += '_weekavg'
    elif use_biweekly_stock:
        db_table += '_biweekly'

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {db_table} WHERE method = '{method}'", conn)
    global_vals.engine_ali.dispose()
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y-%m-%d')
    field_col = df.columns.to_list()[2:-1]

    if change:  # calculate the change of original ratio from T-1 -> T0
        df[field_col] = df[field_col]/df.sort_values(['period_end']).groupby(['group'])[field_col].shift(1)-1

    df.columns = df.columns.to_list()[:2] + ['org_'+x for x in field_col] + [df.columns.to_list()[-1]]

    return df.iloc[:,:-1]

def combine_data(use_biweekly_stock, stock_last_week_avg):
    ''' combine factor premiums with ratios '''

    # calc_premium_all(stock_last_week_avg, use_biweekly_stock)

    if use_biweekly_stock and stock_last_week_avg:
        raise ValueError("Expecting 'use_biweekly_stock' or 'stock_last_week_avg' is TRUE. Got both is TRUE")

    # Read sql from different tables
    factor_table_name = global_vals.factor_premium_table
    if use_biweekly_stock:
        factor_table_name+='_biweekly'
    elif stock_last_week_avg:
        factor_table_name+='_weekavg'

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM {factor_table_name} WHERE \"group\" IS NOT NULL', conn)
        formula = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
    global_vals.engine_ali.dispose()

    # Research stage using 10 selected factor only
    factors = formula.sort_values(by=['rank']).loc[formula['factors'], 'name'].to_list()         # remove factors no longer used
    x_col = formula.sort_values(by=['rank']).loc[formula['x_col'], 'name'].to_list()         # x_col remove highly correlated variables
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y-%m-%d')                 # convert to datetime

    df = df.loc[df['period_end'] < dt.datetime.today() + MonthEnd(-2)]  # remove records within 2 month prior to today

    # 1. Add Macroeconomic variables - from Datastream
    macros = download_clean_macros(df, use_biweekly_stock)
    x_col.extend(macros.columns.to_list()[1:])              # add macros variables name to x_col

    # 2. Add index return variables
    index_ret = download_index_return(use_biweekly_stock, stock_last_week_avg)
    x_col.extend(index_ret.columns.to_list()[1:])           # add index variables name to x_col

    # 3. Add original ratios variables
    org_df = download_org_ratios(use_biweekly_stock, stock_last_week_avg)

    # Combine non_factor_inputs and move it 1-month later -> factor premium T0 assumes we knows price as at T1
    # Therefore, we should also know other data (macro/index/group fundamental) as at T1
    non_factor_inputs = macros.merge(index_ret, on=['period_end'], how='outer')
    non_factor_inputs = org_df.merge(non_factor_inputs, on=['period_end'], how='outer')
    if use_biweekly_stock:
        non_factor_inputs['period_end'] = non_factor_inputs['period_end'].apply(lambda x: x-relativedelta(weeks=2))
    else:
        non_factor_inputs['period_end'] = non_factor_inputs['period_end'] + MonthEnd(-1)

    df = df.merge(non_factor_inputs, on=['group', 'period_end'], how='left')

    print('      ------------------------> Factors: ', factors)

    return df.sort_values(by=['group', 'period_end']), factors, x_col

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self, use_biweekly_stock=False, stock_last_week_avg=False):
        ''' combine all possible data to be used '''

        # define self objects
        self.sample_set = {}
        self.group = pd.DataFrame()
        self.main, self.factor_list, self.original_x_col = combine_data(use_biweekly_stock, stock_last_week_avg)    # combine all data

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

        current_x_col = []
        current_group = self.group.copy(1)

        # Calculate the time_series history for predicted Y (use 1/2/12 based on ARIMA results)
        for i in [1, 2, 12]:
            ar_col = [f"ar_{x}_{i}m" for x in y_type]
            current_group[ar_col] = current_group.groupby(['group'])[y_type].shift(i)
            current_x_col.extend(ar_col)    # add AR variables name to x_col

        # Calculate the moving average for predicted Y
        ma_col = [f"ma_{x}_{ma_period}m" for x in y_type]
        current_group.loc[:, ma_col] = current_group.groupby(['group'])[y_type].transform(
            lambda x: x.rolling(ma_period, min_periods=6).mean())
        current_x_col.extend(ma_col)        # add MA variables name to x_col

        y_col = ["y_" + x for x in y_type]

        # split training/testing sets based on testing_period
        start = testing_period - relativedelta(years=10)    # train df = 40 quarters
        self.train = current_group.loc[(start <= current_group['period_end']) &
                                     (current_group['period_end'] < testing_period)]
        self.train = self.train.dropna(subset=y_col).reset_index(drop=True)      # remove training sample with NaN Y

        self.test = current_group.loc[current_group['period_end'] == testing_period].reset_index(drop=True)

        self.y_qcut_all()   # qcut/cut for all factors to be predicted at the same time

        def divide_set(df):
            ''' split x, y from main '''
            x_col = self.original_x_col + current_x_col + ["org_"+x for x in y_type]
            y_col_cut = [x+'_cut' for x in y_col]
            return df.filter(x_col).values, df[y_col].values, df[y_col_cut].values, df.filter(x_col).columns.to_list()     # Assuming using all factors

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
    y_type = ['vol_30_90', 'tax_less_pension_to_accu_depre', 'book_to_price', 'fwd_ey', 'stock_return_r6_2', 'gross_margin', 'market_cap_usd', 'cash_ratio']
    group_code = 'industry'

    data = load_data(use_biweekly_stock=False, stock_last_week_avg=True)
    data.split_group(group_code)
    sample_set, cv = data.split_all(testing_period, y_type=y_type)
    print(data.x_col)

    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)