import pandas as pd
import os
from general.utils import to_excel

def read_top_excel():
    path = os.getcwd()
    files = os.listdir(path)
    files_xls = [f for f in files if (f[-4:] == 'xlsx')&(f[:3]=='top')]
    # files_xls = [f for f in files if (f[-4:] == 'xlsx')&(f[:3]=='top')&(f[-6]=="-")]

    df_all = []
    for f in files_xls:
        print(f)
        for i in [10, 20]:
            try:
                df = pd.read_excel(f, f"Top {i} Picks(agg)")
                df["n_pick"] = i
                df["n_config"] = int(f.split('_')[0][3:])
                df["since_year"] = int(f.split('_')[1])
                df["weeks_to_expire"] = int(f.split('_')[2][1:])
                df["use_usd"] = ('usd' in f.split('_')[-1])
                df["uid"] = f.split('_')[-3] if ('usd' in f.split('_')[-1]) else f.split('_')[-2]
                df_all.append(df)
            except Exception as e:
                print(e)

    sort_col = ['currency_code', 'weeks_to_expire', 'n_pick', 'n_config', 'since_year', 'use_usd']
    df_all = pd.concat(df_all, axis=0).sort_values(by=sort_col, ascending=False)

    df_all = df_all.loc[df_all["uid"]=="20220220200030"]

    xls_sheets = {}
    for name, g in df_all.groupby(['n_pick']):
        # for k in ["positive_pct", "return"]:
        #     for i in ['weeks_to_expire', 'n_config', 'since_year', 'use_usd']:
        #         results_df = g.pivot(index=[x for x in sort_col if x not in ['currency_code', i]],
        #                              columns=['currency_code', i],
        #                              values=k)
        #         results_df_mean = pd.DataFrame(results_df.mean()).transpose()
        #         results_df_mean.index = [(0, 0, 0, 0)]
        #         results_df = results_df.append(results_df_mean)
        #         results_df.columns = ['-'.join([str(e) for e in x]) for x in results_df]
        #         xls_sheets[f'{i}-{k}'] = results_df.reset_index()

        # SUM1: by configs
        r = g.groupby(['currency_code', 'n_config']).mean()
        # r.columns = ['-'.join([str(e) for e in x]) for x in r]
        xls_sheets[f'configs'] = r.reset_index()

        # SUM2: given config=10: by use_usd
        g1 = g.loc[g['n_config']==10]
        r = pd.pivot_table(g1, index=['currency_code', 'weeks_to_expire','since_year'], columns=['use_usd'], values="return")
        # r.columns = ['-'.join([str(e) for e in x]) for x in r]
        xls_sheets[f'use_usd'] = r.reset_index()

        # SUM2: given config=10, best_use_usd: by since_year
        s = [{"currency_code": "HKD", "use_usd": False},
             {"currency_code": "USD", "use_usd": True},
             {"currency_code": "EUR", "use_usd": True},
             {"currency_code": "HKD", "use_usd": False}]

        g2 = g1.loc[((g1["currency_code"]!="CNY")&g1["use_usd"])|((g1["currency_code"]=="CNY")&~g1["use_usd"])]
        r = pd.pivot_table(g2, index=['currency_code', 'weeks_to_expire'], columns=['since_year'], values=["positive_pct", "return"])
        r.columns = ['-'.join([str(e) for e in x]) for x in r]
        xls_sheets[f'since_year'] = r.reset_index()

        to_excel(xls_sheets, file_name=(f"backtest_analysis_{name}_currency"))
    print(df_all)



if __name__ == '__main__':
    read_top_excel()