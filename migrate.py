from utils import models, read_query, upsert_data_to_database
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import func, select
import os


os.environ['DEBUG'] = 'False'

data = read_query(select(models.Universe))

os.environ['DEBUG'] = 'True'
upsert_data_to_database(
    data, table=models.Universe.__tablename__)
