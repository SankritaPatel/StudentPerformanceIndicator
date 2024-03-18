import os
import sys
from src.StudentPerformanceIndicator.exception import CustomException
from src.StudentPerformanceIndicator.logger import logging
import pandas as pd
import pymysql
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL Database Started...")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established:",mydb)
        df = pd.read_sql_query('select * from student', mydb)
        print(df.head())
        return df
    except Exception as e:
        raise CustomException(e, sys)

