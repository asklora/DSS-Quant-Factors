import numpy as np
import pandas as pd
import datetime as dt
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

from global_vars import *
from general.sql_process import read_table, read_query

def add_arr_col(df, arr, col_name):
    add_df = pd.DataFrame(arr, columns=col_name)
    return pd.concat([df.reset_index(drop=True), add_df], axis=1)

def download_clean_macros():
    ''' download macros data from DB and preprocess: convert some to yoy format '''

    logging.info(f'Download macro data from {macro_data_table}')

    # combine macros & vix data
    macros = read_table(macro_data_table, db_url_read)
    vix = read_table(vix_data_table, db_url_read)
    vix = vix.rename(columns={"vix_id":"field", "vix_value":"value"})
    macros = macros.append(vix)

    macros = macros.pivot(index=["trading_day"], columns=["field"], values="value").reset_index()
    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')

    yoy_col = macros.select_dtypes('float').columns[macros.select_dtypes('float').mean(axis=0) > 100]  # convert YoY
    num_col = macros.select_dtypes('float').columns.to_list()  # all numeric columns

    # update yoy ratios
    macros_yoy = macros[["trading_day"]+list(yoy_col)]
    macros_yoy["trading_day"] = macros_yoy["trading_day"].apply(lambda x: x-relativedelta(years=1))
    macros = macros.merge(macros_yoy, on=["trading_day"], how="outer", suffixes=("","_1yb"))
    macros = macros.sort_values(by="trading_day").fillna(method='ffill')
    for i in yoy_col:
        macros[i] = macros[i]/macros[i+"_1yb"] - 1
    macros = macros.dropna(subset=num_col, how="all")

    return macros[["trading_day"]+list(num_col)]

def download_index_return():
    ''' download index return data from DB and preprocess: convert to YoY and pivot table '''

    logging.info(f'Download index return data from [{processed_ratio_table}]')

    # read stock return from ratio calculation table
    index_query = f"SELECT * FROM {processed_ratio_table} WHERE ticker like '.%%'"
    index_ret = read_query(index_query, db_url_read)
    index_ret = index_ret.pivot(index=["ticker","trading_day"], columns=["field"], values="value").reset_index()

    # Index using all index return12_7, return6_2 & vol_30_90 for 6 market based on num of ticker
    major_index = ['trading_day','.SPX','.CSI300','.SXXGR']    #  try include 3 major market index first
    index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]

    index_col = set(index_ret.columns.to_list()) & {'stock_return_ww1_0', 'stock_return_r6_2'}
    index_ret = index_ret.set_index(['trading_day', 'ticker'])[list(index_col)].unstack()
    index_ret.columns = [f'{x[1]}_{x[0]}' for x in index_ret.columns.to_list()]
    index_ret = index_ret.reset_index()
    index_ret['trading_day'] = pd.to_datetime(index_ret['trading_day'])

    return index_ret

def combine_data(weeks_to_expire, average_days, update_since=None, mode='v2'):
    ''' combine factor premiums with ratios '''

    # calc_premium_all(stock_last_week_avg, use_biweekly_stock)

    # Read premium sql from different tables
    factor_table_name = factor_premium_table

    logging.info(f'Use [{weeks_to_expire}] week premium')
    conditions = ['"group" IS NOT NULL',
                  f"weeks_to_expire={weeks_to_expire}",
                  f"average_days={average_days}"]
    
    if isinstance(update_since, datetime):
        update_since_str = update_since.strftime(r'%Y-%m-%d %H:%M:%S')
        conditions.append(f"trading_day >= TO_TIMESTAMP('{update_since_str}', 'YYYY-MM-DD HH:MI:SS')")

    prem_query = f'SELECT * FROM {factor_table_name} WHERE {" AND ".join(conditions)};'
    df = read_query(prem_query, db_url_read)
    df = df.pivot(index=['trading_day', 'group'], columns=['field'], values="value")

    if mode == 'trim':
        df = df.filter(regex='^trim_')
    elif mode == 'v2':
        df = df.drop(regex='^trim_')

    df.columns.name = None
    df = df.reset_index()
    df['trading_day'] = pd.to_datetime(df['trading_day'], format='%Y-%m-%d')  # convert to datetime

    # read formula table
    formula = read_table(formula_factors_table_prod, db_url_read)
    formula = formula.loc[formula['name'].isin(df.columns.to_list())]       # filter existing columns from factors

    # Research stage using 10 selected factor only
    x_col = {}
    x_col['factor'] = formula['name'].to_list()         # x_col remove highly correlated variables

    for p in formula['pillar'].unique():
        x_col[p] = formula.loc[formula['pillar']==p, 'name'].to_list()         # factor for each pillar

    # df = df.loc[df['trading_day'] < dt.datetime.today() + MonthEnd(-2)]  # remove records within 2 month prior to today

    # 1. Add Macroeconomic variables - from Datastream
    macros = download_clean_macros()
    x_col['macro'] = macros.columns.to_list()[1:]              # add macros variables name to x_col

    # 2. Add index return variables
    index_ret = download_index_return()
    x_col['index'] = index_ret.columns.to_list()[1:]           # add index variables name to x_col

    # Combine non_factor_inputs and move it 1-month later -> factor premium T0 assumes we knows price as at T1
    # Therefore, we should also know other data (macro/index/group fundamental) as at T1
    index_ret["trading_day"] = pd.to_datetime(index_ret["trading_day"])
    macros["trading_day"] = pd.to_datetime(macros["trading_day"])
    non_factor_inputs = macros.merge(index_ret, on=['trading_day'], how='outer')
    non_factor_inputs['trading_day'] = non_factor_inputs['trading_day'].apply(lambda x: x-relativedelta(weeks=weeks_to_expire))
    df = df.merge(non_factor_inputs, on=['trading_day'], how='outer').sort_values(['group','trading_day'])
    df[x_col['macro']+x_col['index']] = df.sort_values(["trading_day", "group"]).groupby(["group"])[x_col['macro']+x_col['index']].ffill()
    df = df.dropna(subset=["group"])

    # use only period_end date
    indexes = pd.MultiIndex.from_product([df['group'].unique(), df['trading_day'].unique()],
                                         names=['group', 'trading_day']).to_frame().reset_index(drop=True)
    df = pd.merge(df, indexes, on=['group', 'trading_day'], how='right')
    logging.info(f"Factors: {x_col['factor']}")

    return df.sort_values(by=['group', 'trading_day']), x_col['factor'], x_col

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self, weeks_to_expire, average_days, update_since=None, mode=''):
        ''' combine all possible data to be used 
        
        Parameters
        ----------
        weeks_to_expire : text
        update_since : bool, optional
        mode : {''(default), 'trim'}, optional

        '''

        # define self objects
        self.sample_set = {}
        self.group = pd.DataFrame()
        self.weeks_to_expire = weeks_to_expire
        self.main, self.factor_list, self.x_col_dict = combine_data(weeks_to_expire, average_days, update_since=update_since, mode=mode)    # combine all data

        # calculate y for all factors
        all_y_col = ["y_"+x for x in self.x_col_dict['factor']]
        self.main[all_y_col] = self.main.groupby(['group'])[self.x_col_dict['factor']].shift(-1)

    def split_group(self, group_name=None):
        ''' split main sample sets in to industry_parition or country_partition '''

        curr_list = ['KRW','GBP','HKD','EUR','CNY','USD','TWD','JPY','SGD'] #
        # curr_list = ['GBP','HKD','EUR','USD'] #

        self.group_name = group_name

        if group_name == 'currency':
            self.group = self.main.loc[self.main['group'].isin(curr_list)]          # train on industry partition factors
        elif group_name == 'industry':
            self.group = self.main.loc[~self.main['group'].str.len()!=3]          # train on currency partition factors
        elif group_name in curr_list:
            self.group = self.main.loc[self.main['group']==group_name]
        else:
            self.group = self.main.loc[self.main['group'].isin(group_name.split(','))]

    def y_replace_median(self, qcut_q, arr, arr_cut, arr_test, arr_test_cut):
        ''' convert qcut results (e.g. 012) to the median of each group for regression '''

        df = pd.DataFrame(np.vstack((arr, arr_cut))).T   # concat original & qcut
        median = df.groupby([1]).median().sort_index()[0].to_list()     # find median of each group
        arr_cut_median = pd.DataFrame(arr_cut).replace(range(qcut_q), median)[0].values
        arr_test_cut_median = pd.DataFrame(arr_test_cut).replace(range(qcut_q), median)[0].values
        return arr_cut_median, arr_test_cut_median

    def neg_factor_best_period(self, df, x_col):

        best_best = {}
        for name in x_col:
            best = {}
            g = df[name]
            for i in np.arange(12, 120, 12):
                g['ma'] = g.rolling(i, min_periods=1, closed='left')['premium'].mean()
                g['new_premium'] = np.where(g['ma'] >= 0, g['premium'], -g['premium'])
                best[i] = g['new_premium'].mean()

            best_best[name] = [k for k, v in best.items() if v == np.max(list(best.values()))][0]

        return best_best

    def y_qcut_all(self, qcut_q, defined_cut_bins, use_median, y_type, use_average=False):
        ''' convert continuous Y to discrete (0, 1, 2) for all factors during the training / testing period '''

        null_col = self.train.isnull().sum()
        null_col = list(null_col.loc[(null_col == len(self.train))].index)  # remove null col from y col
        y_col = ['y_' + x for x in y_type if x not in null_col]
        cut_col = [x + "_cut" for x in y_col]

        # convert consistently negative premium factor to positive
        self.__y_convert_neg_factors(use_average=use_average)

        # use n-split qcut median as Y
        if qcut_q > 0:
            arr = self.train[y_col].values.flatten()  # Flatten all training factors to qcut all together
            # arr[(arr>np.quantile(np.nan_to_num(arr), 0.99))|(arr<np.quantile(np.nan_to_num(arr), 0.01))] = np.nan

            if defined_cut_bins == []:
                # cut original series into bins
                arr_cut, cut_bins = pd.qcut(arr, q=qcut_q, retbins=True, labels=False)
                cut_bins[0], cut_bins[-1] = [-np.inf, np.inf]
            else:
                # use pre-defined cut_bins for cut (since all factor should use same cut_bins)
                cut_bins = defined_cut_bins
                arr_cut = pd.cut(arr, bins=cut_bins, labels=False)

            arr_test = self.test[y_col].values.flatten()  # Flatten all testing factors to qcut all together
            arr_test_cut = pd.cut(arr_test, bins=cut_bins, labels=False)

            if use_median:      # for regression -> remove noise by regression on median of each bins
                arr_cut, arr_test_cut = self.y_replace_median(qcut_q, arr, arr_cut, arr_test, arr_test_cut)

            self.train[cut_col] = np.reshape(arr_cut, (len(self.train), len(y_col)), order='C')
            self.test[cut_col] = np.reshape(arr_test_cut, (len(self.test), len(y_col)), order='C')

            # write cut_bins to DB
            self.cut_bins_df = self.train[cut_col].apply(pd.value_counts).transpose()
            self.cut_bins_df = self.cut_bins_df.divide(self.cut_bins_df.sum(axis=1).values, axis=0).reset_index()
            self.cut_bins_df['negative'] = False
            # self.cut_bins_df.loc[self.cut_bins_df['index'].isin([x+'_cut' for x in neg_factor]), 'negative'] = True
            self.cut_bins_df['y_type'] = [x[2:-4] for x in self.cut_bins_df['index']]
            self.cut_bins_df['cut_bins_low'] = cut_bins[1]
            self.cut_bins_df['cut_bins_high'] = cut_bins[-2]
        else:
            self.train[cut_col] = self.train[y_col]
            self.test[cut_col] = self.test[y_col]

        return y_col

    def __y_convert_neg_factors(self, use_average=False, n_years=7, lasso_alpha=1e-5):
        '''  convert consistently negative premium factor to positive
        refer to: https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/994803719/AI+Score+Brainstorms+2022-01-28

        Parameters
        ----------
        use_average (Bool, Optional):
            if True, use training period average; else (default) use lasso
        n_years (Int, Optional):
            lasso training period length (default = 7) years
        lasso_alpha (Float, Optional):
            lasso training alpha lasso_alpha (default = 1e-5)
        '''

        y_col = read_query(f'SELECT name FROM {formula_factors_table_prod} WHERE is_active')['name'].to_list()

        if use_average:         # using average of training period -> next period +/-
            m = self.train.filter(y_col).mean(axis=0)
            self.neg_factor = list(m[m<0].index)
        else:
            start_date = dt.datetime.today() - relativedelta(years=n_years)
            n_x = len(self.train.loc[self.train['trading_day']>=start_date, 'trading_day'].unique())

            train_X = self.train.set_index('trading_day').filter(y_col).stack().reset_index()
            train_X.columns = ['trading_day', 'field', 'y']

            for i in range(1, n_x + 1):
                train_X[f'x_{i}'] = train_X.groupby(['field'])['y'].shift(i)
            train_X = train_X.dropna(how='any')
            clf = Lasso(alpha=lasso_alpha)
            clf.fit(train_X.filter(regex='^x_').values, train_X['y'].values)

            test_X = train_X.groupby('field').last().reset_index()
            test_X['pred'] = clf.predict(test_X.filter(regex='^x_|^y$').values[:,:-1])
            self.neg_factor = test_X.loc[test_X['pred'] < 0, 'field'].to_list()

        self.neg_factor = []
        self.train[self.neg_factor + ['y_'+x for x in self.neg_factor]] *= -1
        self.test[self.neg_factor + ['y_'+x for x in self.neg_factor]] *= -1

    def split_train_test(self, testing_period, output_options, input_options):
        ''' split training / testing set based on testing period '''

        current_group = self.group.copy(1)
        start = testing_period - relativedelta(years=20)    # train df = 20*12 months

        # factor with ARMA history as X5
        arma_col = self.x_col_dict['factor'] + self.x_col_dict['index'] + self.x_col_dict['macro'] # if using pca, all history first

        # 1. [Prep X] Calculate the time_series history for predicted Y (use 1/2/12 based on ARIMA results)
        self.x_col_dict['ar'] = []
        for i in input_options["ar_period"]:
            ar_col = [f"ar_{x}_{i}m" for x in arma_col]
            current_group[ar_col] = current_group.groupby(['group'])[arma_col].shift(i)
            self.x_col_dict['ar'].extend(ar_col)      # add AR variables name to x_col

        # 2. [Prep X] Calculate the moving average for predicted Y
        self.x_col_dict['ma'] = []
        for i in input_options["ma3_period"]:     # include moving average of 3-5, 6-8, 9-11
            ma_q = current_group.groupby(['group'])[arma_col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            ma_q_col = ma_q.columns = [f"ma_{x}_q" for x in arma_col]
            current_group = pd.concat([current_group, ma_q], axis=1)
            current_group[[f'{x}{i}' for x in ma_q_col]] = current_group.groupby(['group'])[ma_q_col].shift(i)
            self.x_col_dict['ma'].extend([f'{x}{i}' for x in ma_q_col])  # add MA variables name to x_col
        for i in input_options["ma12_period"]:          # include moving average of 12 - 23
            ma_y = current_group.groupby(['group'])[arma_col].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
            ma_y_col = ma_y.columns = [f"ma_{x}_y" for x in arma_col]
            current_group = pd.concat([current_group, ma_y], axis=1)
            current_group[[f'{x}{i}' for x in ma_y_col]] = current_group.groupby(['group'])[ma_y_col].shift(i)
            self.x_col_dict['ma'].extend([f'{x}{i}' for x in ma_y_col])        # add MA variables name to x_col

        # 3. [Split training/testing] sets based on testing_period
        self.train = current_group.loc[(start <= current_group['trading_day']) &
               (current_group['trading_day'] <= (testing_period-relativedelta(weeks=self.weeks_to_expire)))].copy()
        self.test = current_group.loc[current_group['trading_day'] == testing_period].reset_index(drop=True).copy()

        # 4. [Prep Y]: qcut/cut for all factors to be predicted (according to factor_formula table in DB) at the same time
        self.y_col = self.y_qcut_all(**output_options)
        self.train = self.train.dropna(subset=self.y_col, how='any').reset_index(drop=True)      # remove training sample with NaN Y

        # if using feature selection with PCA
        arma_factor = [x for x in self.x_col_dict['ar']+self.x_col_dict['ma'] for f in self.x_col_dict['factor'] if f in x]

        # 5. [Prep X] use PCA on all Factor + ARMA inputs
        factor_pca_col = self.x_col_dict['factor']+arma_factor
        factor_pca_train, factor_pca_test, factor_feature_name = load_data.standardize_pca_x(
            self.train[factor_pca_col], self.test[factor_pca_col], input_options["factor_pca"])
        self.x_col_dict['arma_pca'] = ['arma_'+str(x) for x in factor_feature_name]
        self.train[self.x_col_dict['arma_pca']] = factor_pca_train
        self.test[self.x_col_dict['arma_pca']] = factor_pca_test
        logging.info(f"After {input_options['factor_pca']} PCA [Factors]: {len(factor_feature_name)}")

        # 6. [Prep X] use PCA on all index/macro inputs
        group_index = {"USD":".SPX", "HKD":".HSI", "EUR":".SXXGR"}
        mi_pca_col = [x for x in self.x_col_dict['index'] if re.match(f'^{group_index[self.group_name]}', x)]
        mi_pca_col += self.x_col_dict['macro']
        mi_pca_train, mi_pca_test, mi_feature_name = load_data.standardize_pca_x(
            self.train[mi_pca_col], self.test[mi_pca_col], input_options["mi_pca"])
        self.x_col_dict['mi_pca'] = ['mi_'+str(x) for x in mi_feature_name]
        self.train[self.x_col_dict['mi_pca']] = mi_pca_train
        self.test[self.x_col_dict['mi_pca']] = mi_pca_test
        logging.info(f"After {input_options['mi_pca']} PCA [Macro+Index]: {len(mi_feature_name)}")

        def divide_set(df):
            ''' split x, y from main '''
            x_col = self.x_col_dict['arma_pca'] + self.x_col_dict['mi_pca']
            y_col_cut = [x+'_cut' for x in self.y_col]
            return df.filter(x_col).values, np.nan_to_num(df[self.y_col].values, 0), np.nan_to_num(df[y_col_cut].values), \
                   df.filter(x_col).columns.to_list()     # Assuming using all factors

        self.sample_set['train_x'], self.sample_set['train_y'], self.sample_set['train_y_final'],_ = divide_set(self.train)
        self.sample_set['test_x'], self.sample_set['test_y'], self.sample_set['test_y_final'], self.x_col = divide_set(self.test)

    @staticmethod
    def standardize_pca_x(X_train, X_test, n_components=None):
        ''' standardize x + PCA applied to x with train_x fit '''

        org_feature = X_train.columns.to_list()
        X_train, X_test = X_train.values, X_test.values
        if n_components:
            scaler = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
            X_train = np.nan_to_num(X_train, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)
        else:
            scaler = StandardScaler()

        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)
        if n_components:
            feature_name = range(1, x_train.shape[1]+1)
        else:
            feature_name = org_feature
        return x_train, x_test, feature_name

    def split_valid(self, testing_period, n_splits, valid_method):
        ''' split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set

        Parameters
        ----------
        testing_period :
            testing set starting date.
        n_splits (Float):
            if > 1, n * (train + valid) sets split from the last 2 years of training;
            if < 1, use last n_splits% of training as validation.
        valid_method (Str):
            if "cv", cross validation between groups (n_splits must > 1);
            if "chron", use last 2 years / % of training as validation.
        '''

        if valid_method == "cv":       # split validation set by cross-validation 5 split
            gkf = GroupShuffleSplit(n_splits=n_splits, random_state=666).split(self.sample_set['train_x'],
                                                                              self.sample_set['train_y_final'],
                                                                              groups=self.train['group'])
        elif valid_method == "chron":       # split validation set by chronological order
            gkf = []
            if n_splits >= 1:
                for n in range(1, n_splits+1):
                    valid_period = testing_period - relativedelta(days=round(365*2/n_splits*n))   # using last 2 year samples as valid set
                    valid_index = self.train.loc[self.train['trading_day'] >= valid_period].index.to_list()
                    train_index = self.train.loc[self.train['trading_day'] < valid_period].index.to_list()
                    gkf.append((train_index, valid_index))
            elif n_splits > 0:
                valid_len = (self.train['trading_day'].max() - self.train['trading_day'].min())*n_splits
                valid_period = self.train['trading_day'].max() - valid_len
                valid_index = self.train.loc[self.train['trading_day'] >= valid_period].index.to_list()
                train_index = self.train.loc[self.train['trading_day'] < valid_period].index.to_list()
                gkf.append((train_index, valid_index))
            else:
                raise ValueError("Invalid 'n_splits'. Expecting 'n_splits' > 0 got ", n_splits)
        elif isinstance(valid_method, int):
            # valid_method can be year name (e.g. 2010) -> use n_splits% amount of data since 2010-01-01 as valid sets
            assert n_splits < 1
            valid_len = (self.train['trading_day'].max() - self.train['trading_day'].min()) * n_splits
            valid_start = dt.datetime(valid_method, 1, 1, 0, 0, 0)
            valid_end = valid_start + valid_len
            valid_index = self.train.loc[(self.train['trading_day'] >= valid_start) &
                                         (self.train['trading_day'] < valid_end)].index.to_list()
            train_index = self.train.loc[(self.train['trading_day'] < (valid_start - valid_len/2)) |     # half of valid sample have data leak from training sets
                                         (self.train['trading_day'] >= valid_end)].index.to_list()
            gkf = [(train_index, valid_index)]
        else:
            raise ValueError("Invalid 'valid_method'. Expecting 'cv' or 'chron' or Int of year (e.g. 2010) got ", valid_method)
        return gkf

    def split_all(self, testing_period, n_splits=5, valid_method='cv',
                  output_options={"y_type": None, "qcut_q": 10, "use_median": False, "defined_cut_bins": []},
                  input_options={"ar_period": [1,2], "ma3_period": [3, 6, 9], "ma12_period": [12], "factor_pca": 0.6, "mi_pca": 0.9}):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, output_options, input_options)   # split x, y for test / train samples
        gkf = self.split_valid(testing_period, n_splits, valid_method)           # split for cross validation in groups
        return self.sample_set, gkf

if __name__ == '__main__':

    testing_period = dt.datetime(2021,12,26)
    group_code = 'USD'

    data = load_data(weeks_to_expire=4, average_days=7)
    y_type = data.factor_list[:5]  # random forest model predict all factor at the same time

    data.split_group(group_code)

    # for y in y_type:
    sample_set, cv = data.split_all(testing_period, valid_method='chron', n_splits=0.2,
                                    output_options={"y_type": y_type, "qcut_q": 10, "use_median": False,
                                                    "defined_cut_bins": []},
                                    input_options={"ar_period": [], "ma3_period": [], "ma12_period": [],
                                                   "factor_pca": 0.6, "mi_pca": 0.9})
    # print(data.cut_bins)

    print(data.x_col)

    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)