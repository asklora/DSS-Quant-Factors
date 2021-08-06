import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from preprocess.premium_calculation import calc_premium_all, trim_outlier
import global_vals
# from scipy.fft import fft, fftfreq, rfft, rfftfreq
from sqlalchemy import text


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
        db_table_name = global_vals.processed_ratio_table + '_weekavg'
    else:
        db_table_name = global_vals.processed_ratio_table

    with global_vals.engine_ali.connect() as conn:
        index_ret = pd.read_sql(f"SELECT * FROM {db_table_name} WHERE ticker like '.%%'", conn)
    global_vals.engine_ali.dispose()

    # Index using all index return12_7, return6_2 & vol_30_90 for 6 market based on num of ticker
    major_index = ['period_end','.SPX','.CSI300','.SXXGR']    # try include 3 major market index first
    index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]
    index_ret = index_ret.set_index(['period_end', 'ticker'])[['stock_return_r12_7','stock_return_r6_2', 'vol_30_90']].unstack()
    index_ret.columns = [f'{x[1]}_{x[0][0]}{x[0][-1]}' for x in index_ret.columns.to_list()]
    index_ret = index_ret.reset_index()
    index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])

    return index_ret

def download_org_ratios(use_biweekly_stock, stock_last_week_avg, method='median', change=True):
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
        df[field_col] = df[field_col].apply(trim_outlier)

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

    # df = df.loc[df['period_end'] < dt.datetime.today() + MonthEnd(-2)]  # remove records within 2 month prior to today

    # 1. Add Macroeconomic variables - from Datastream
    macros = download_clean_macros(df, use_biweekly_stock)
    x_col.extend(macros.columns.to_list()[1:])              # add macros variables name to x_col

    # 2. Add index return variables
    index_ret = download_index_return(use_biweekly_stock, stock_last_week_avg)
    x_col.extend(index_ret.columns.to_list()[1:])           # add index variables name to x_col

    # Combine non_factor_inputs and move it 1-month later -> factor premium T0 assumes we knows price as at T1
    # Therefore, we should also know other data (macro/index/group fundamental) as at T1
    non_factor_inputs = macros.merge(index_ret, on=['period_end'], how='outer')
    if use_biweekly_stock:
        non_factor_inputs['period_end'] = non_factor_inputs['period_end'].apply(lambda x: x-relativedelta(weeks=2))
    else:
        non_factor_inputs['period_end'] = non_factor_inputs['period_end'] + MonthEnd(-1)

    df = df.merge(non_factor_inputs, on=['period_end'], how='left').sort_values(['group','period_end'])

    # 3. (Removed) Add original ratios variables
    org_df = download_org_ratios(use_biweekly_stock, stock_last_week_avg)
    df = df.merge(org_df, on=['group', 'period_end'], how='left')

    # make up for all missing date in df
    indexes = pd.MultiIndex.from_product([df['group'].unique(), df['period_end'].unique()], names=['group', 'period_end']).to_frame().reset_index(drop=True)
    df = pd.merge(df, indexes, on=['group', 'period_end'], how='right')
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

        # calculate y for all factors
        self.all_y_col = ["y_"+x for x in self.factor_list]
        self.main[self.all_y_col] = self.main.groupby(['group'])[self.factor_list].shift(-1)

    def split_group(self, group_name=None):
        ''' split main sample sets in to industry_parition or country_partition '''

        curr_list = ['TWD','JPY','KRW','GBP','HKD','SGD','EUR','CNY','USD']

        if group_name == 'currency':
            self.group = self.main.loc[self.main['group'].isin(curr_list)]          # train on industry partition factors
        else:
            self.group = self.main.loc[~self.main['group'].isin(curr_list)]          # train on currency partition factors

        self.cross_factors = self.important_cross_factor(group_name)

    def important_cross_factor(self, group_name):
        ''' figure the top 3 most important cross factors '''

        with global_vals.engine_ali.connect() as conn:
            query = text(
                f"SELECT P.*, S.group_code, S.testing_period, S.y_type FROM {global_vals.feature_importance_table}_lgbm_class P "
                f"INNER JOIN {global_vals.result_score_table}_lgbm_class S ON S.finish_timing = P.finish_timing "
                f"WHERE S.name_sql='biweekly_ma'")
            df = pd.read_sql(query, conn)  # download training history
            formula = pd.read_sql(f'SELECT name, rank, x_col FROM {global_vals.formula_factors_table}', conn)
        global_vals.engine_ali.dispose()

        # filter cross factors only
        x_col = formula.sort_values(by=['rank']).loc[formula['x_col'], 'name'].to_list()
        df = df.loc[df['name'].isin(x_col)]

        # find the most important factor for industry / currency partition
        df1 = df.loc[df['group_code'] == group_name].groupby(['name', 'y_type'])['split'].mean().unstack()
        dic = {}
        for col in df1.columns.to_list():
            dic[col] = list(df1[col].sort_values(ascending=False).index)[:4]

        return dic

    def corr_cross_factor(self, df, y_type):
        ''' figure the top 3 most important cross factors '''

        df = df[self.original_x_col].corr()[y_type]
        dic = {y_type:list(df.sort_values(ascending=False).index)[:10]}
        print('Correlation ', dic)
        return dic

    def y_replace_median(self, qcut_q, arr, arr_cut):
        ''' convert qcut results (e.g. 012) to the median of each group for regression '''

        df = pd.DataFrame(np.vstack((arr, arr_cut))).T   # concat original & qcut
        median = df.groupby([1]).median().sort_index()[0].to_list()     # find median of each group
        arr_cut_median = pd.DataFrame(arr_cut).replace(range(qcut_q), median)[0].values
        return arr_cut_median

    def y_qcut_all(self, qcut_q, defined_cut_bins, use_median):
        ''' convert continuous Y to discrete (0, 1, 2) for all factors during the training / testing period '''

        cut_col = [x + "_cut" for x in self.all_y_col]

        if qcut_q > 0:
            arr = self.train[self.all_y_col].values.flatten()  # Flatten all training factors to qcut all together
            # arr[(arr>np.quantile(np.nan_to_num(arr), 0.99))|(arr<np.quantile(np.nan_to_num(arr), 0.01))] = np.nan

            if defined_cut_bins == []:
                # cut original series into bins
                arr_cut, self.cut_bins = pd.qcut(arr, q=qcut_q, retbins=True, labels=False)
                # arr, cut_bins = pd.cut(arr, bins=3, retbins=True, labels=False)
                self.cut_bins[0], self.cut_bins[-1] = [-np.inf, np.inf]
            else:
                # use pre-defined cut_bins for cut (since all factor should use same cut_bins)
                self.cut_bins = defined_cut_bins
                arr_cut = pd.cut(arr, bins=self.cut_bins, labels=False)

            arr_test = self.test[self.all_y_col].values.flatten()  # Flatten all testing factors to qcut all together
            arr_test_cut = pd.cut(arr_test, bins=self.cut_bins, labels=False)

            if use_median:      # for regression -> remove noise by regression on median of each bins
                arr_cut = self.y_replace_median(qcut_q, arr, arr_cut)
                arr_test_cut = self.y_replace_median(qcut_q, arr_test, arr_test_cut)

            self.train[cut_col] = np.reshape(arr_cut, (len(self.train), len(self.all_y_col)), order='C')
            self.test[cut_col] = np.reshape(arr_test_cut, (len(self.test), len(self.all_y_col)), order='C')
        else:
            self.train[cut_col] = self.train[self.all_y_col]
            self.test[cut_col] = self.test[self.all_y_col]
            self.cut_bins = ''

    def split_train_test(self, testing_period, y_type, qcut_q, defined_cut_bins, use_median):
        ''' split training / testing set based on testing period '''

        current_x_col = []
        current_group = self.group.copy(1)
        start = testing_period - relativedelta(years=8)    # train df = 40 quarters

        corr_df = current_group.loc[(start <= current_group['period_end']) & (current_group['period_end'] < testing_period)]
        self.cross_factors = self.corr_cross_factor(corr_df, y_type[0])

        # 1. Calculate the time_series history for predicted Y (use 1/2/12 based on ARIMA results)
        for i in [1,2]:
            if len(y_type) == 1:
                ar_col_org = self.cross_factors[y_type[0]]        # for LGBM: add AR for top 4 factors based on importance
            else:
                ar_col_org = y_type            # for RF: add AR for all y_type predicted at the same time
            ar_col = [f"ar_{x}_{i}m" for x in ar_col_org]
            current_group[ar_col] = current_group.groupby(['group'])[ar_col_org].shift(i)
            current_x_col.extend(ar_col)    # add AR variables name to x_col

        # 2. Calculate the moving average for predicted Y
        ma_q_col = [f"ma_{x}_q" for x in y_type]
        ma_y_col = [f"ma_{x}_y" for x in y_type]
        current_group[ma_q_col] = current_group.groupby(['group'])[y_type].rolling(3, min_periods=1).mean().values      # moving average for 3m
        current_group[ma_y_col] = current_group.groupby(['group'])[y_type].rolling(12, min_periods=1).mean().values      # moving average for 12m
        if len(y_type) == 1:        # not for RF -> avoid too many inputs
            for i in [3, 6, 9]:     # include moving average of 3-5, 6-8, 9-11
                current_group[[f'{x}{i}' for x in ma_q_col]] = current_group.groupby(['group'])[ma_q_col].shift(i)
                current_x_col.extend([f'{x}{i}' for x in ma_q_col])  # add MA variables name to x_col
        for i in [12]:          # include moving average of 12 - 23
            current_group[[f'{x}{i}' for x in ma_y_col]] = current_group.groupby(['group'])[ma_y_col].shift(i)
            current_x_col.extend([f'{x}{i}' for x in ma_y_col])        # add MA variables name to x_col

        y_col = ['y_'+x for x in y_type]

        # split training/testing sets based on testing_period
        self.train = current_group.loc[(start <= current_group['period_end']) & (current_group['period_end'] < testing_period)]
        self.test = current_group.loc[current_group['period_end'] == testing_period].reset_index(drop=True)

        # qcut/cut for all factors to be predicted (according to factor_formula table in DB) at the same time
        self.y_qcut_all(qcut_q, defined_cut_bins, use_median)

        self.train = self.train.dropna(subset=y_col).reset_index(drop=True)      # remove training sample with NaN Y

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

    def split_valid(self, testing_period, n_splits, valid_method):
        ''' split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set '''

        if valid_method == "cv":       # split validation set by cross-validation 5 split
            gkf = GroupShuffleSplit(n_splits=n_splits, random_state=666).split( self.sample_set['train_x'],
                                                                              self.sample_set['train_y_final'],
                                                                              groups=self.train['group'])
        elif valid_method == "chron":       # split validation set by chronological order
            gkf = []
            for n in range(1, n_splits+1):
                valid_period = testing_period - relativedelta(days=round(365*2/n_splits*n))   # using last 2 year samples as valid set
                test_index = self.train.loc[self.train['period_end'] >= valid_period].index.to_list()
                train_index = self.train.loc[self.train['period_end'] < valid_period].index.to_list()
                gkf.append((train_index, test_index))
        else:
            raise ValueError("Invalid valid_method. Expecting 'cv' or 'chron' got ", valid_method)

        return gkf

    def split_all(self, testing_period, y_type, qcut_q=3, n_splits=5, valid_method='cv',
                  defined_cut_bins=[], use_median=False):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, y_type, qcut_q, defined_cut_bins, use_median)   # split x, y for test / train samples
        self.standardize_x()                                          # standardize x array
        gkf = self.split_valid(testing_period, n_splits, valid_method)           # split for cross validation in groups

        return self.sample_set, gkf

if __name__ == '__main__':
    fft_combine()
    exit(1)
    # download_org_ratios('mean')
    # download_index_return()
    testing_period = dt.datetime(2021,5,23)
    y_type = ['tax_less_pension_to_accu_depre', 'book_to_price', 'fwd_ey', 'stock_return_r6_2', 'gross_margin', 'market_cap_usd', 'cash_ratio']
    group_code = 'industry'

    data = load_data(use_biweekly_stock=True, stock_last_week_avg=False)

    data.split_group(group_code)

    for y in y_type:
        sample_set, cv = data.split_all(testing_period, y_type=[y], use_median=True)
        print(data.cut_bins)

    print(data.x_col)

    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)