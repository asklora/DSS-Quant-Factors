import global_vals
import pandas as pd

with global_vals.engine_ali.connect() as conn:
    df = pd.read_sql(f"SELECT * FROM {global_vals.processed_pca_table}", conn)
global_vals.engine_ali.dispose()

df = pd.groupby(['testing_period','group','index'])