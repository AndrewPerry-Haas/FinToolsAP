"""
This is an example file that is used to add a file to the database
where either the functionality to read that type of file into the 
database is not present OR because a certain file wont read into the 
database properly. 

A table normally fails being read into the database because it has 
a mixed type in one of the columns causing the error. So files in 
the ExtraScripts/ folder are usefull when trying to cast the entire
dataframe to one type.

For example, the IBES Long Term Growth dataset (fpi = 0) from WRDS
cannot be loaded into the dataframe because it has a mixed type.
So, I use an "ExtraScript" to cast the entire dataframe into a 
string type and load it into the datframe from this file. Below 
is the example of how to handle the situation described above.

YOU MUST SPECIFY A PATH TO THE FILE YOU ARE TRYING TO LOAD

Best Practices:
    - Name the file the same name as the resulting table
    - Dont forget to update the DatabaseParameters file
    - Put the file you are trying to cast into the 
        FilestoSQL/ directory so it is easy to understand 
        what data is in the databse

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

import pandas

PATH_TO_STUBBORN_FILE = pathlib.Path('~/Documents/FullDB/FILEStoDB')
ltg = pandas.read_csv(PATH_TO_STUBBORN_FILE / 'IBES_LTG.csv', low_memory = False)
ltg = ltg.astype(str)
ltg.to_sql('IBES_LTG', con = SQL_ENGINE, index = False)