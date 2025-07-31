"""
This script explains how to read and write pandas or polars 
dataframes to the SQL database. The file creates a connection 
to a SQLite database and provides functions for reading and 
writing data using both pandas and polars libraries.

Usage:
    This file will be automatically called by FinToolsAP.LocalDatabase
    on initalization, if the table this file creates is not in
    the database and the name of this file is in 
    DatabaseParameters.Tables.CREATED_TABLES list.

Attributes:
    PATH_TO_DB (str): The path to the SQLite database file.
    PATH_TO_DBP (pathlib.Path): The path to the DatabaseContents.py file.
    DB_CONNECTION (str): The connection string for the SQLite database.
    SQL_ENGINE (sqlalchemy.engine.Engine): The SQL engine for writing pandas DataFrame.

Dependencies:
    - sys
    - pathlib
    - sqlalchemy
    - importlib.util
    - polars
    - pandas
    - connectorx

Functions:
    - connectorx.read_sql: Read data into a pandas or polars DataFrame using connectorx.
    - pandas.read_sql: Read data into a pandas DataFrame using pandas (not recommended).
    - polars.write_database: Write a polars.DataFrame to the SQLite database.

Best Practices:
    - Name the file the same name as the resulting table
    - Dont forget to update the DatabaseParameters file
    - Only read data that is currently in the database.
        Dont read data from external directories. This
        makes keeping track of the data present to the DB 
        much simpler. The data used to create tables should 
        be 1. present in the database and 2. in its raw form.
        This makes it easier to track down bugs that might
        exist in code.

"""

## DO NOT MODIFY

import sys
import pathlib
import sqlalchemy
import importlib.util

PATH_TO_DB = sys.argv[1]
PATH_TO_DBP = pathlib.Path(PATH_TO_DB).parent / 'DatabaseContents.py'

spec = importlib.util.spec_from_file_location('DBP', str(PATH_TO_DBP))
DBP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DBP)

# connection for connectorx (reading) and polars.write_database
# for polars.DataFrame.
DB_CONNECTION = f'sqlite:///{PATH_TO_DB}'

# sql engine for writing pandas.DataFrame. Additionally, 
# pandas.read_sql can be used for reading from the database
# using the sql engine. However this it is slower and more
# memeory inefficient than connectorx
SQL_ENGINE = sqlalchemy.create_engine(DB_CONNECTION)

############################################################################
# Your Code Below

import polars
import pandas
import connectorx


## Working with pandas 

# Reading
# read data into a pandas dataframe using connectorx (Recomended)
df = connectorx.read_sql(conn = DB_CONNECTION, 
                         query = """YOUR SQLITE QUERY STRING"""
                        )

# read date into a pandas dataframe using pandas (Not Recomended)
df = pandas.read_sql(sql = """YOUR SQLITE QUERY STRING""", 
                     con = SQL_ENGINE
                    )

# Writing
# write dataframe to database
df.to_sql(name = 'TABLE NAME',
          con = SQL_ENGINE,
          index = False
        )

## Working with polars

# Reading
df = connectorx.read_sql(conn = DB_CONNECTION, 
                         query = 'YOUR SQLITE QUERY STRING',
                         return_type = 'polars'
                        )

# Writing
df.write_database(table_name = 'TABLE NAME',
                  connection = DB_CONNECTION
                )

