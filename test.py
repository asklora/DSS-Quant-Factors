from general.sql_process import read_query, upsert_data_to_database, uid_maker
import pandas as pd

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



