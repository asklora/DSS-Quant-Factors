from dotenv import load_dotenv
from environs import Env

env = Env()
load_dotenv()

import os

DEBUG = os.getenv("DEBUG").lower() == 'true'
