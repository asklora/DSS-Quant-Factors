import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import text
from dateutil.relativedelta import relativedelta

from global_vals import engine_ali, result_score_table
from hyperspace_lgbm import find_hyperspace

# find hyper-parameters used in hyperopt
sql_result = {'objective': 'multiclass', 'group_code':'industry'}
params = list(find_hyperspace(sql_result).keys())

# params = ['num_leaves', 'min_data_in_leaf']
matrix = 'accuracy'

r_name = 'biweekly'
iter_name = r_name.split('_')[-1]
y_type = 'market_cap_usd'

def download_from_score():

    with engine_ali.connect() as conn:
        query = text(f"select * "
                     f"FROM (SELECT DISTINCT *, max({matrix}_valid) OVER (partition by group_code, testing_period, "
                     f"cv_number, y_type) as max_thing FROM {result_score_table}_lgbm_class WHERE name_sql = '{r_name}') t "
                     f"where {matrix}_valid = max_thing")
        results = pd.read_sql(query, con=conn)  # read records from DB
    engine_ali.dispose()

    results.to_csv(f'tuning_origin_{iter_name}.csv')

    return results

def calc_correl():
    ''' calculated correlation between train/valid/test sets '''

    results = download_from_score()
    
    correls = {}
    correls['train_valid'] = {}
    correls['valid_test'] = {}

    correls['train_valid']['all'] = results[f'{matrix}_train'].corr(results[f'{matrix}_valid'])
    correls['valid_test']['all'] = results[f'{matrix}_valid'].corr(results[f'{matrix}_test'])

    for i in set(results['group_code']):
        part = results.loc[results['group_code']==i]
        correls['train_valid'][i] = part[f'{matrix}_train'].corr(part[f'{matrix}_valid'])
        correls['valid_test'][i] = part[f'{matrix}_valid'].corr(part[f'{matrix}_test'])

    print(pd.DataFrame(correls))
    pd.DataFrame(correls).to_csv(f'tuning_correl_{iter_name}.csv')

def calc_average():
    ''' calculate mean of each variable in db '''
    
    results = download_from_score()

    writer = pd.ExcelWriter(f'tuning/tuning_avg_test_{iter_name}.xlsx')    # create excel records

    for c in set(results['group_code']):
        sub_df = results.loc[results['group_code']==c]

        df_list = []
        for p in params:
            try:    # calculate means of each subset
                des_df = sub_df.groupby(p).mean()[[f'{matrix}_test',f'r2_test']].reset_index()
                des_df['len'] = sub_df.groupby(p).count().reset_index()[f'{matrix}_test']
                des_df.columns = ['subset', f'{matrix}_test', f'r2_test', 'len']
                des_df['params'] = p
                des_df = des_df.sort_values(by=[f'{matrix}_test'], ascending=True)
                df_list.append((des_df))
            except:
                continue
        pd.concat(df_list, axis=0).to_excel(writer, f'{c}', index=False)

    writer.save()

def plot_scatter_single_param():
    ''' plot a scatter map of average results in DB '''

    ratio = f'{matrix}_test'
    df = download_from_score().drop_duplicates(['y_type', ratio])
    all_y_type = list(set(df['y_type']))

    for p in params:
        k=1
        fig = plt.figure(figsize=(2 * len(all_y_type), 10), dpi=120, constrained_layout=True)
        for name, g in df.groupby('group_code'):
            ax = fig.add_subplot(2, 1, k)
            max_scores = pd.DataFrame(g.groupby(['y_type',p]).mean()[ratio], columns=[ratio])
            max_scores.loc[:, 'len'] = g.groupby(['y_type',p]).count()[ratio].to_list()
            ax.scatter([x[0] for x in max_scores.index], [x[1] for x in max_scores.index],
                       c=max_scores[ratio], s=max_scores['len'] / len(g) * 30000, cmap='coolwarm')
            for i in range(len(max_scores)):
                ax.annotate(max_scores.iloc[i,-2].round(2), (max_scores.index[i][0], max_scores.index[i][1]), fontsize=15)
            ax.set_ylabel(name, fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=15)
            k+=1

        plt.suptitle(p, fontsize=20)
        fig.savefig(f'tuning/tuning_plot_{iter_name}_[{p}].png')
        plt.close()

def plot_scatter():
    ''' plot a scatter map of average results in DB '''
    
    df = download_from_score()

    n = len(params)
    ratio = f'{matrix}_test'

    for y in list(set(df['y_type'])):
        results = df[y]
        for name, g in results.groupby(['group_code']):
            print(name)
            fig = plt.figure(figsize=(4 * n, 4 * n), dpi=120)  # create figure for test & train boxplot

            k = 1
            for p1 in params:
                for p2 in params:
                    ax = fig.add_subplot(n, n, k)
                    max_scores = pd.DataFrame(g.groupby([p1,p2]).mean()[ratio], columns=[ratio])
                    max_scores.loc[:, 'len'] = g.groupby([p1,p2]).count()[ratio].to_list()
                    ax.scatter([x[0] for x in max_scores.index], [x[1] for x in max_scores.index],
                               c=max_scores[ratio], s=max_scores['len']/len(g)*1000, cmap='coolwarm')
                    ax.set_xlabel(p1)
                    ax.set_ylabel(p2)
                    k+=1

            fig.tight_layout()
            fig.savefig(f'tuning/tuning_plot_test_{iter_name}_{name}.png')
            plt.close()

if __name__ == "__main__":

    # calc_correl(matrix=m)                                # check correlation
    calc_average()
    # plot_scatter()
    # plot_scatter_single_param()