import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn import linear_model

import global_vars
from general.sql_output import sql_read_table, sql_read_query

def add_arr_col(df, arr, col_name):
    add_df = pd.DataFrame(arr, columns=col_name)
    return pd.concat([df.reset_index(drop=True), add_df], axis=1)

def download_clean_macros(main_df):
    ''' download macros data from DB and preprocess: convert some to yoy format '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download macro data from {global_vars.macro_data_table}')

    macros = sql_read_table(global_vars.macro_data_table, global_vars.db_url_aws_read)
    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')

    yoy_col = macros.select_dtypes('float').columns[macros.select_dtypes('float').mean(axis=0) > 100]  # convert YoY
    num_col = macros.select_dtypes('float').columns.to_list()  # all numeric columns

    macros[yoy_col] = (macros[yoy_col] / macros[yoy_col].shift(12)).sub(1)  # convert yoy_col to YoY

    # combine macros & vix data
    macros["period_end"] = pd.to_datetime(macros["trading_day"])
    # print(num_col)

    # create map for macros (currency + type of data)
    macro_map = pd.DataFrame()
    macro_map['name'] = num_col[:-2] # except fred_data / data columns
    macro_map['group'] = macro_map['name'].str[:2].replace(['us','jp','ch','em'],['USD','JPY','CNY','EUR'])
    macro_map['type'] = macro_map['name'].str[2:].replace(['inter3','gbill','mshort'],['ibor3','gbond','ibor3'])

    # add vix to macro_data
    df_date_list = main_df['period_end'].drop_duplicates().sort_values()
    macros = macros.merge(pd.DataFrame(df_date_list.values, columns=['period_end']), on=['period_end'],
                          how='outer').sort_values(['period_end'])
    macros = macros.fillna(method='ffill')
    macros = macros.loc[macros['period_end'].isin(df_date_list)]

    return macros.drop(['trading_day'], axis=1)

def download_index_return(tbl_suffix):
    ''' download index return data from DB and preprocess: convert to YoY and pivot table '''

    print(f'#################################################################################################')
    print(f'      ------------------------> Download index return data from {global_vars.processed_ratio_table}')

    # read stock return from ratio calculation table
    index_query = f"SELECT * FROM {global_vars.processed_ratio_table}{tbl_suffix} WHERE ticker like '.%%'"
    index_ret = sql_read_query(index_query, global_vars.db_url_alibaba_prod)

    # Index using all index return12_7, return6_2 & vol_30_90 for 6 market based on num of ticker
    major_index = ['period_end','.SPX','.CSI300','.SXXGR']    # try include 3 major market index first
    index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]

    index_col = set(index_ret.columns.to_list()) & {'stock_return_ww1_0', 'stock_return_ww2_1', 'stock_return_ww4_2', 'stock_return_r12_7','stock_return_r6_2'}
    index_ret = index_ret.set_index(['period_end', 'ticker'])[list(index_col)].unstack()
    index_ret.columns = [f'{x[1]}_{x[0][0]}{x[0][-1]}' for x in index_ret.columns.to_list()]
    index_ret = index_ret.reset_index()
    index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])

    return index_ret

def combine_data(tbl_suffix, update_since=None, mode='v2'):
    ''' combine factor premiums with ratios '''

    # calc_premium_all(stock_last_week_avg, use_biweekly_stock)

    # Read sql from different tables
    factor_table_name = global_vars.factor_premium_table

    print(f'      ------------------------> Use {tbl_suffix} ratios')
    conditions = ['"group" IS NOT NULL']
    
    if isinstance(update_since, datetime):
        update_since_str = update_since.strftime(r'%Y-%m-%d %H:%M:%S')
        conditions.append(f"period_end >= TO_TIMESTAMP('{update_since_str}', 'YYYY-MM-DD HH:MI:SS')")

    if mode == 'v2_trim':
        conditions.append('trim_outlier')
    elif mode == 'v2':
        conditions.append('not trim_outlier')
    else:
        raise Exception('Unknown mode')

    prem_query = f'SELECT period_end, "group", factor_name, premium FROM {factor_table_name}{tbl_suffix}_{mode} WHERE {" AND ".join(conditions)};'
    df = sql_read_query(prem_query, global_vars.db_url_alibaba_prod)
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y-%m-%d')  # convert to datetime
    df = df.pivot(['period_end', 'group'], ['factor_name']).droplevel(0, axis=1)
    df.columns.name = None
    df = df.reset_index()

    formula = sql_read_table(global_vars.formula_factors_table_prod, global_vars.db_url_alibaba_prod)
    formula = formula.loc[formula['name'].isin(df.columns.to_list())]       # filter existing columns from factors

    # Research stage using 10 selected factor only
    x_col = {}
    x_col['factor'] = formula['name'].to_list()         # x_col remove highly correlated variables

    for p in formula['pillar'].unique():
        x_col[p] = formula.loc[formula['pillar']==p, 'name'].to_list()         # factor for each pillar

    # df = df.loc[df['period_end'] < dt.datetime.today() + MonthEnd(-2)]  # remove records within 2 month prior to today

    # 1. Add Macroeconomic variables - from Datastream
    macros = download_clean_macros(df)
    x_col['macro'] = macros.columns.to_list()[1:]              # add macros variables name to x_col

    # 2. Add index return variables
    index_ret = download_index_return(tbl_suffix)
    x_col['index'] = index_ret.columns.to_list()[1:]           # add index variables name to x_col

    # Combine non_factor_inputs and move it 1-month later -> factor premium T0 assumes we knows price as at T1
    # Therefore, we should also know other data (macro/index/group fundamental) as at T1
    non_factor_inputs = macros.merge(index_ret, on=['period_end'], how='outer')
    if 'weekly' in tbl_suffix:
        non_factor_inputs['period_end'] = non_factor_inputs['period_end'].apply(lambda x: x-relativedelta(weeks=int(tbl_suffix[-1])))
    elif 'monthly' in tbl_suffix:
        non_factor_inputs['period_end'] = non_factor_inputs['period_end'].apply(lambda x: x - relativedelta(months=int(tbl_suffix[-1])))
    df = df.merge(non_factor_inputs, on=['period_end'], how='left').sort_values(['group','period_end'])

    # make up for all missing date in df
    indexes = pd.MultiIndex.from_product([df['group'].unique(), df['period_end'].unique()], names=['group', 'period_end']).to_frame().reset_index(drop=True)
    df = pd.merge(df, indexes, on=['group', 'period_end'], how='right')
    print('      ------------------------> Factors: ', x_col['factor'])

    return df.sort_values(by=['group', 'period_end']), x_col['factor'], x_col

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self, tbl_suffix, update_since=None, mode='default'):
        ''' combine all possible data to be used 
        
        Parameters
        ----------
        tbl_suffix : text
        update_since : bool, optional
        mode : {default 'default', 'v2', 'v2_trim'}, optional
        
        '''

        # define self objects
        self.sample_set = {}
        self.group = pd.DataFrame()
        self.main, self.factor_list, self.x_col_dict = combine_data(
            tbl_suffix,
            update_since=update_since,
            mode=mode)    # combine all data

        # calculate y for all factors
        all_y_col = ["y_"+x for x in self.x_col_dict['factor']]
        self.main[all_y_col] = self.main.groupby(['group'])[self.x_col_dict['factor']].shift(-1)
        # print(self.main)

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

        print(self.group)

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

    def y_qcut_all(self, qcut_q, defined_cut_bins, use_median, test_change, y_col):
        ''' convert continuous Y to discrete (0, 1, 2) for all factors during the training / testing period '''

        null_col = self.train.isnull().sum()
        null_col = list(null_col.loc[(null_col == len(self.train))].index)  # remove null col from y col
        y_col = ['y_' + x for x in y_col if x not in null_col]
        cut_col = [x + "_cut" for x in y_col]

        # convert consistently negative premium factor to positive
        m = self.train[y_col].mean(axis=0)
        self.neg_factor = list(m[m<0].index)

        # neg_factor = self.x_col_dict['neg_factor']
        self.train[self.neg_factor] = -self.train[self.neg_factor]
        self.test[self.neg_factor] = -self.test[self.neg_factor]

        if qcut_q > 0:

            arr = self.train[y_col].values.flatten()  # Flatten all training factors to qcut all together
            # arr[(arr>np.quantile(np.nan_to_num(arr), 0.99))|(arr<np.quantile(np.nan_to_num(arr), 0.01))] = np.nan

            if defined_cut_bins == []:
                # cut original series into bins
                arr_cut, cut_bins = pd.qcut(arr, q=qcut_q, retbins=True, labels=False)
                # arr, cut_bins = pd.cut(arr, bins=3, retbins=True, labels=False)
                cut_bins[0], cut_bins[-1] = [-np.inf, np.inf]
            else:
                # use pre-defined cut_bins for cut (since all factor should use same cut_bins)
                cut_bins = defined_cut_bins
                arr_cut = pd.cut(arr, bins=cut_bins, labels=False)

            arr_test = self.test[y_col].values.flatten()  # Flatten all testing factors to qcut all together
            arr_test_cut = pd.cut(arr_test, bins=cut_bins, labels=False)

            if use_median:      # for regression -> remove noise by regression on median of each bins
                arr_cut, arr_test_cut = self.y_replace_median(qcut_q, arr, arr_cut, arr_test, arr_test_cut)

            arr_add = np.reshape(arr_cut, (len(self.train), len(y_col)), order='C')
            self.train = add_arr_col(self.train, arr_add, cut_col)
            arr_add = np.reshape(arr_test_cut, (len(self.test), len(y_col)), order='C')
            self.test = add_arr_col(self.test, arr_add, cut_col)

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

    def split_train_test(self, testing_period, y_type, qcut_q, defined_cut_bins, use_median, test_change, use_pca,
                         write_pca_component=False):
        ''' split training / testing set based on testing period '''

        current_group = self.group.copy(1)
        start = testing_period - relativedelta(years=20)    # train df = 20*12 months

        # factor with ARMA history as X5
        arma_col = self.x_col_dict['factor'] + self.x_col_dict['index'] + self.x_col_dict['macro'] # if using pca, all history first

        # 1. Calculate the time_series history for predicted Y (use 1/2/12 based on ARIMA results)
        self.x_col_dict['ar'] = []
        for i in [1,2]:
            ar_col = [f"ar_{x}_{i}m" for x in arma_col]
            current_group[ar_col] = current_group.groupby(['group'])[arma_col].shift(i)
            self.x_col_dict['ar'].extend(ar_col)      # add AR variables name to x_col

        # 2. Calculate the moving average for predicted Y
        ma_q = current_group.groupby(['group'])[arma_col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        ma_y = current_group.groupby(['group'])[arma_col].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
        ma_q_col = ma_q.columns = [f"ma_{x}_q" for x in arma_col]
        ma_y_col = ma_y.columns = [f"ma_{x}_y" for x in arma_col]
        current_group = pd.concat([current_group, ma_q, ma_y], axis=1)
        self.x_col_dict['ma'] = []
        for i in [3, 6, 9]:     # include moving average of 3-5, 6-8, 9-11
            current_group[[f'{x}{i}' for x in ma_q_col]] = current_group.groupby(['group'])[ma_q_col].shift(i)
            self.x_col_dict['ma'].extend([f'{x}{i}' for x in ma_q_col])  # add MA variables name to x_col
        for i in [12]:          # include moving average of 12 - 23
            current_group[[f'{x}{i}' for x in ma_y_col]] = current_group.groupby(['group'])[ma_y_col].shift(i)
            self.x_col_dict['ma'].extend([f'{x}{i}' for x in ma_y_col])        # add MA variables name to x_col

        y_col = ['y_'+x for x in y_type]

        # split training/testing sets based on testing_period
        # self.train = current_group.loc[(current_group['period_end'] < testing_period)].copy()
        self.train = current_group.loc[(start <= current_group['period_end'].dt.date) &
                                       (current_group['period_end'].dt.date < testing_period)].copy()
        self.test = current_group.loc[current_group['period_end'].dt.date == testing_period].reset_index(drop=True).copy()

        # qcut/cut for all factors to be predicted (according to factor_formula table in DB) at the same time
        self.y_col = y_col = self.y_qcut_all(qcut_q, defined_cut_bins, use_median, test_change, y_type)
        # self.train = self.train.dropna(subset=y_col, how='all').reset_index(drop=True)      # remove training sample with NaN Y
        self.train = self.train.dropna(subset=y_col, how='any').reset_index(drop=True)      # remove training sample with NaN Y

        if use_pca>0.1:  # if using feature selection with PCA

            arma_factor = [x for x in self.x_col_dict['ar']+self.x_col_dict['ma'] for f in self.x_col_dict['factor'] if f in x]
            arma_mi = [x for x in self.x_col_dict['ar']+self.x_col_dict['ma'] for f in self.x_col_dict['index'] + self.x_col_dict['macro'] if f in x]

            # use PCA on all ARMA inputs
            arma_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=use_pca))])
            pca_arma_df = self.train[self.x_col_dict['factor']+arma_factor].fillna(0)
            arma_pca = arma_pipe.fit(pca_arma_df)
            arma_trans = arma_pca.transform(pca_arma_df)
            self.x_col_dict['arma_pca'] = [f'arma_{i}' for i in range(1, arma_trans.shape[1]+1)]
            print(f"      ------------------------> After {use_pca} PCA [Factors]: {len(self.x_col_dict['arma_pca'])}")

            # write PCA components to DB
            # if write_pca_component:
            #     df = pd.DataFrame(arma_pca.components_, index=self.x_col_dict['arma_pca'], columns=self.x_col_dict['factor'] + arma_factor).reset_index()
            #     df['var_ratio'] = np.cumsum(arma_pca.explained_variance_ratio_)
            #     df['group'] = self.group_name
            #     df['testing_period'] = testing_period
            #     with global_vars.engine_ali.connect() as conn:
            #         extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 1000}
            #         conn.execute(f"DELETE FROM {global_vars.processed_pca_table} "
            #                      f"WHERE testing_period='{dt.datetime.strftime(testing_period, '%Y-%m-%d')}'")   # remove same period prediction if exists
            #         df.to_sql(global_vars.processed_pca_table, **extra)
            #         pd.DataFrame({global_vars.processed_pca_table: {'update_time': dt.datetime.now()}}).reset_index().to_sql(global_vars.update_time_table, **extra)
            #     global_vars.engine_ali.dispose()

            self.train = add_arr_col(self.train, arma_trans, self.x_col_dict['arma_pca'])
            arr = arma_pca.transform(self.test[self.x_col_dict['factor']+arma_factor].fillna(0))
            self.test = add_arr_col(self.test, arr, self.x_col_dict['arma_pca'])

            # use PCA on all index/macro inputs
            arma_mi = []
            mi_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.6))])
            pca_mi_df = self.train[self.x_col_dict['index']+self.x_col_dict['macro']+arma_mi].fillna(-1)
            mi_pca = mi_pipe.fit(pca_mi_df)
            mi_trans = mi_pca.transform(pca_mi_df)
            self.x_col_dict['mi_pca'] = [f'mi_{i}' for i in range(1, mi_trans.shape[1]+1)]
            print(f"      ------------------------> After 0.8 PCA [Macros]: {len(self.x_col_dict['mi_pca'])}")

            self.train = add_arr_col(self.train, mi_trans, self.x_col_dict['mi_pca'])
            arr = mi_pca.transform(self.test[self.x_col_dict['index']+self.x_col_dict['macro']+arma_mi].fillna(-1))
            self.test = add_arr_col(self.test, arr, self.x_col_dict['mi_pca'])

        elif use_pca>0:     # if using feature selection with LASSO (alpha=l1)
            all_input = self.x_col_dict['factor']+self.x_col_dict['ar']+self.x_col_dict['ma'] + \
                        self.x_col_dict['index']+self.x_col_dict['macro']
            pca_arma_df = StandardScaler().fit_transform(self.train[all_input].fillna(0))
            pca_arma_df_y = np.nan_to_num(self.train[y_col].values,0)
            w = np.array(range(len(pca_arma_df))) / len(pca_arma_df)
            w = np.tanh(w - 0.5) + 0.5
            arma_pca = linear_model.Lasso(alpha=use_pca).fit(pca_arma_df, pca_arma_df_y, sample_weight=w)
            self.x_col_dict['arma_pca'] = list(np.array(all_input)[np.sum(arma_pca.coef_, axis=0)!=0])
            print(f"      ------------------------> After {use_pca} PCA [Factors]: {len(self.x_col_dict['arma_pca'])}")
            self.x_col_dict['mi_pca'] = []

        def divide_set(df):
            ''' split x, y from main '''
            x_col = self.x_col_dict['factor'] + self.x_col_dict['ar'] + self.x_col_dict['ma'] + self.x_col_dict['macro'] + self.x_col_dict['index']
            if use_pca>0:
                x_col = self.x_col_dict['arma_pca'] + self.x_col_dict['mi_pca']#+['fred_data','usgdp','usinter3']
            y_col_cut = [x+'_cut' for x in y_col]

            return df.filter(x_col).values, np.nan_to_num(df[y_col].values,0), np.nan_to_num(df[y_col_cut].values), \
                   df.filter(x_col).columns.to_list()     # Assuming using all factors

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
                test_index = self.train.loc[self.train['period_end'].dt.date >= valid_period].index.to_list()
                train_index = self.train.loc[self.train['period_end'].dt.date < valid_period].index.to_list()
                gkf.append((train_index, test_index))
        else:
            raise ValueError("Invalid valid_method. Expecting 'cv' or 'chron' got ", valid_method)

        return gkf

    def split_all(self, testing_period, y_type, qcut_q=3, n_splits=5, valid_method='cv',
                  defined_cut_bins=[], use_median=False, test_change=False, use_pca=0):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, y_type, qcut_q, defined_cut_bins, use_median, test_change=test_change, use_pca=use_pca)   # split x, y for test / train samples
        self.standardize_x()                                          # standardize x array
        gkf = self.split_valid(testing_period, n_splits, valid_method)           # split for cross validation in groups
        return self.sample_set, gkf

if __name__ == '__main__':
    # fft_combine()
    # exit(1)
    # download_org_ratios('mean')
    # download_index_return()
    testing_period = dt.datetime(2021,9,5)
    group_code = 'USD'

    data = load_data(tbl_suffix='_weekly4', mode='v2')
    y_type = data.factor_list  # random forest model predict all factor at the same time

    data.split_group(group_code)

    # for y in y_type:
    sample_set, cv = data.split_all(testing_period, y_type=y_type, use_median=False, valid_method='chron', use_pca=0.6)
    # print(data.cut_bins)

    print(data.x_col)

    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)