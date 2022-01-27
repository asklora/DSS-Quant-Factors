import datetime as dt
import pandas as pd
from general.sql_process import upsert_data_to_database

log_table = "factor_model_log"

def get_timestamp_now_str():
    ''' return timestamp in form of string of numbers '''
    return str(dt.datetime.now()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')

def model_log(log_time=None):
    ''' keep log of iteration config '''

    if type(log_time)==type(None):
        log_time = get_timestamp_now_str()

    details = input("Details: ")

    df = pd.DataFrame([{"log_time": log_time, "details": details}], index=[0])
    upsert_data_to_database(df, log_table, primary_key=["log_time"], how="update")
    print(f"Save log for [{log_time}]")

if __name__ == '__main__':
    model_log()
