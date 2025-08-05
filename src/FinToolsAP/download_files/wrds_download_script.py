# DO NOT MODIFY

import sys
import pathlib
import sqlalchemy

PATH_TO_DB = sys.argv[1]
WRDS_USERN = sys.argv[2]
STR_TABLES = sys.argv[3]

# add parent directory of this script to sys.path
parent_dir = pathlib.Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
import _config

# add DatabaseContents to sys.path
sys.path.append(str(pathlib.Path(PATH_TO_DB).parent))
import DatabaseContents as DBC

sql_engine = sqlalchemy.create_engine('sqlite:///' + str(PATH_TO_DB))

############################################################################
# Your Code Below

import wrds
import time

missing_tables = STR_TABLES.split(',')
WRDS_db = wrds.Connection(username = WRDS_USERN)
for table_name in missing_tables:
    table = table_name.replace('_', '.', 1)
    print(f'{_config.bcolors.INFO}Downloading {table} from WRDS...{_config.bcolors.ENDC}')
    s = time.time()
    sql_str = f"""SELECT * FROM {table}"""
    # download the data to a dataframe
    df = WRDS_db.raw_sql(sql_str)
    # write the dataframe to the local sql database
    df.to_sql(table_name, con = sql_engine, if_exists = 'replace', index = False)
    e = time.time()
    print(f'{_config.bcolors.OK}Finished {table_name}: {round(e - s, 3)}s.{_config.bcolors.ENDC}')
