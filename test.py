from general.sql_process import read_query, upsert_data_to_database, uid_maker
import pandas as pd
import numpy as np

query = "SELECT * FROM universe_rating_posneg_factor_history"

# logger.info(query)
df = read_query(query.replace(",)", ")"))
df['trading_day'] = pd.to_datetime(df['trading_day'])
df = df.set_index('trading_day').groupby(['ticker', 'weeks_to_expire'], as_index=False).resample('D').pad()
df = df.reset_index().drop(columns="level_0")

upsert_data_to_database(df, "universe_rating_posneg_factor_history", primary_key=['ticker', 'weeks_to_expire',
                                                                                  'trading_day'], how="ignore")

exit(2000)



df = pd.DataFrame({1: {"testing_period": "a", "group": "b", "factor_name": ["c", "d"]},
                   2: {"testing_period": "a2", "group": "b2", "factor_name": ["c0", "c1", "d1"]}}).transpose()
df['len'] = df['factor_name'].str.len()

a = df[["testing_period", "group"]].values
b = df['len'].values

arr1 = np.repeat(df[["testing_period", "group"]].values, df['len'].values, axis=0)
arr2 = np.array([e for x in df["factor_name"].to_list() for e in x])[:, np.newaxis]

df_new = np.concatenate([arr1, arr2], axis=1)
df_new = pd.DataFrame(df.columns.to_list())
print(df)


exit(1)

df = pd.read_excel('macro_key.xlsx', sheet_name='Sheet1')
df['Region'] = df['Region'].replace({"M#USKEY": "USD",
                                     "M#EMKEY": "EUR",
                                     "M#CHKEY": "CNY",
                                     "M#HKKEY": "HKD"})
df = df[['Region', 'RIC', "ESFREQ"]].rename(columns={"RIC": "dsws_name", "Region": "currency_code"})

df['monthly'] = (df['ESFREQ'] == 'MONTHLY')
df['quarterly'] = (df['ESFREQ'] == 'QUARTERLY')
df['annually'] = (df['ESFREQ'] == 'ANNUALLY')
df = df.drop(columns=['ESFREQ'])
df['is_active'] = False
df['our_name'] = df['dsws_name']
print(df)

df_macro = read_query('SELECT * FROM ingestion_name_macro where is_active')
df_macro['is_active'] = True
df_macro['annually'] = False
print(df_macro)

r = df_macro.set_index(['our_name'])['dsws_name'].to_dict()
data_macro = read_query('SELECT * FROM data_macro_key')
data_macro['field'] = data_macro['field'].replace(r)
data_macro = uid_maker(data_macro, primary_key=['trading_day', 'field'])
upsert_data_to_database(data_macro, 'data_macro', primary_key=['trading_day', 'field'], how='update')

# df_macro = df_macro.append(df)
# df_macro = df_macro.drop_duplicates(["dsws_name"], keep='first')
# print(df_macro.shape)
#
# upsert_data_to_database(df_macro, 'ingestion_name_macro', primary_key=['dsws_name'], how='update')



