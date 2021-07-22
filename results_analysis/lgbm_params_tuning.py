import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import text
from dateutil.relativedelta import relativedelta

from global_vals import engine, result_score_table
from hyperspace_lgbm import find_hyperspace

class params_tuning:

    def __init__(self, r_name):
        ''' donwload results from results_lightgbm '''

        self.r_name = r_name
        
        sql_result = {'objective':'multiclass'}
        space = find_hyperspace(sql_result)                 
        self.params = list(space.keys())             # find hyper-parameters used in hyperopt
    
        with engine.connect() as conn:
            query = text("select * from (select DISTINCT *, min(mse_valid) over (partition by group_code, "
                         "testing_period, cv_number) as min_thing from {} where name_sql in :name) t "
                         "where mse_valid = min_thing".format(result_score_table))
            query = query.bindparams(name=tuple(self.r_name))
            self.results = pd.read_sql(query, con=conn)   # read records from DB
        engine.dispose()

        self.results.to_csv(f'params_tuning_origin_{self.r_name[0][:10]}.csv')

        print(self.results['testing_period'].max())

        last_valid_period = self.results['testing_period'].max() - relativedelta(years=1)
        self.results = self.results.loc[self.results['testing_period']<=last_valid_period]

        print('===== score results =====', self.results)

    def calc_correl(self, matrix='mse'):
        ''' calculated correlation between train/valid/test sets '''
    
        correls = {}
        correls['train_valid'] = {}
        correls['valid_test'] = {}
    
        correls['train_valid']['all'] = self.results[f'{matrix}_train'].corr(self.results[f'{matrix}_valid'])
        correls['valid_test']['all'] = self.results[f'{matrix}_valid'].corr(self.results[f'{matrix}_test'])

        for i in set(self.results['group_code']):
            part = self.results.loc[self.results['group_code']==i]
            correls['train_valid'][i] = part[f'{matrix}_train'].corr(part[f'{matrix}_valid'])
            correls['valid_test'][i] = part[f'{matrix}_valid'].corr(part[f'{matrix}_test'])
    
        print(pd.DataFrame(correls))
        pd.DataFrame(correls).to_csv(f'params_tuning_correl_{self.r_name[0][:10]}.csv')

    def calc_average(self, eval_matrix='mae', eval_sample='test'):
        ''' calculate mean of each variable in db '''
    
        writer = pd.ExcelWriter(f'params_tuning_{eval_sample}_{self.r_name[0][:10]}.xlsx')    # create excel records
    
        for c in set(self.results['group_code']):
            sub_df = self.results.loc[self.results['group_code']==c]

            df_list = []
            for p in self.params:
                try:    # calculate means of each subset
                    des_df = sub_df.groupby(p).mean()[[f'{eval_matrix}_{eval_sample}',f'r2_{eval_sample}']].reset_index()
                    des_df['len'] = sub_df.groupby(p).count().reset_index()[f'{eval_matrix}_{eval_sample}']
                    des_df.columns = ['subset', f'{eval_matrix}_{eval_sample}', f'r2_{eval_sample}', 'len']
                    des_df['params'] = p
                    des_df = des_df.sort_values(by=[f'{eval_matrix}_{eval_sample}'], ascending=True)
                    df_list.append((des_df))
                except:
                    continue
            pd.concat(df_list, axis=0).to_excel(writer, f'{c}', index=False)
    
        writer.save()

    def plot_scatter(self, eval_matrix='mae', eval_sample='test'):
        ''' plot a scatter map of average results in DB '''

        n = len(self.params)
        ratio = f'{eval_matrix}_{eval_sample}'
    
        for name, g in self.results.groupby(['group_code']):
            print(name)
            fig = plt.figure(figsize=(4 * n, 4 * n), dpi=120)  # create figure for test & train boxplot
    
            k = 1
            for p1 in self.params:
                for p2 in self.params:
                    ax = fig.add_subplot(n, n, k)
                    max_scores = pd.DataFrame(g.groupby([p1,p2]).mean()[ratio], columns=[ratio])
                    max_scores.loc[:, 'len'] = g.groupby([p1,p2]).count()[ratio].to_list()
                    ax.scatter([x[0] for x in max_scores.index], [x[1] for x in max_scores.index],
                               c=max_scores[ratio], s=max_scores['len']/len(g)*1000, cmap='coolwarm')
                    ax.set_xlabel(p1)
                    ax.set_ylabel(p2)
                    k+=1
    
            fig.tight_layout()
            fig.savefig(f'params_tuning_plot_{eval_sample}_{self.r_name[0][:10]}_{name}.png')
            plt.close()

if __name__ == "__main__":


    r_name = ['rev_yoy_2021-07-09 09:23:27.029713']
    m = 'mae'

    pt = params_tuning(r_name=r_name)
    pt.calc_correl(matrix=m)                                # check correlation
    pt.calc_average(eval_matrix=m, eval_sample='test')
    pt.plot_scatter(eval_matrix=m, eval_sample='test')
