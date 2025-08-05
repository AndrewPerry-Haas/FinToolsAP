import polars

class Directories:

    # public
    CreateTables = 'CreateTables'
    FILEStoDB = 'FILEStoDB'
    ExtraScripts = 'ExtraScripts'

    # private
    _sub_dirs = [FILEStoDB, CreateTables, ExtraScripts]
    _base_files = 'base_files'
    _download_files = 'download_files'
    _shell_files = 'shell_files'

class RequiredAttributes:

    TABLES = ['EXTRA_TABLES', 'FILES_TABLES', 
              'WRDS_USERNAME', 'WRDS_TABLES', 
              'CREATED_TABLES']

    GENERIC_TABLE = ['TABLE', 'VARS_DATA_TYPE', 
                     'DEFAULT_VARS', 'DEFAULT_DATE', 
                     'DEFAULT_ID']

    RESERVED_ATTR = ['ALL_VARS', 'DEFAULT_STOCK_ID']

class KeywordArguments:

    QUERY_VARS = ['vars', 'add_vars', 'sub_vars', 
                  'all_vars', 'start_date', 'end_date', 
                  'table_info', 'return_type', 'row_limit', 
                  'suppress']

class QueryComponents:

    COMPONENTS = ['table_info', 'start_date', 'end_date', 'vars']

class SQLCommands:

    DROP_TABLE = """DROP TABLE {table}"""

    DROP_NULL_ROW = """DELETE FROM {table} WHERE {col} IS NULL OR trim({col}) = ''"""

    UPPER_CASE_COLUMN = """UPDATE {table} SET {col} = UPPER({col})"""

    SET_NULL_COL = """UPDATE {table} SET {col} = NULL WHERE {col} = ''"""

    SQL_DICT = {'drop_null_row': DROP_NULL_ROW, 
                'upper_col': UPPER_CASE_COLUMN}

    GET_TABLE_NAMES = """SELECT name FROM sqlite_master WHERE type = 'table'"""
    
    CAST_TABLE = (
        """CREATE TABLE {new} AS SELECT {cast} FROM {old}"""
    )

    RENAME_TABLE = (
        """ALTER TABLE {old} RENAME TO {new}"""
    )

class bcolors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OK = '\033[92m'
    INFO = '\033[94m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TypeMapping:

    # maps from pandas types from DatabaseContents to polars types
    TYPE_MAP = {float: polars.Float64,
                str: polars.String,
                int: polars.Int64,
                'datetime64[ns]': polars.Datetime,
                'Int32': polars.Int32,
                'Int64': polars.Int64,
                'Int8': polars.Int8, 
                'Int16': polars.Int16,
                bool: polars.Boolean,
                object: polars.String,
                'object': polars.String}
    
    SQL_TO_PANDAS_TYPE_MAP = {'INTEGER':  '\'Int64\'',
                              'FLOAT':    'float',
                              'REAL':     'float',
                              'TEXT':     'object',
                              'BLOB':     'object',
                              'NULL':     'object',
                              'DATETIME': '\'datetime64[ns]\'',
                              'BIGINT':   '\'Int64\'',
                              'BOOLEAN':  'bool',
                              'DATE':     '\'datetime64[ns]\'',
                              'INT':      '\'Int64\'',
                              'NUM':      'float',
                              'SMALLINT': '\'Int32\''}

class Messages:
    """ Defines custom messages
    """
    ADD_TO_DBC = '{color}Would you like to add the following tables to the DBC file? {tables} [y/n]: ' + bcolors.ENDC
    
    NO_QUERY_VARS = '{color}There are no vars being quired for. Please see if `DEFAULT_VARS` for table {table} is populated. Or use keyword argument `vars`.' + bcolors.ENDC

    COULD_NOT_CAST = '{color}polars could not cast to specified data types. Normally this results from a date like column' + bcolors.ENDC

    INVALID_RETURN_TYPE = '{color}return_type: expected \'pandas\' or \'polars\', got {act}' + bcolors.ENDC

    COLUMN_NOT_CLEANED = '{color}Column {col} in table {tab} was unable to be cleaned. Automatic type-correction may fail.' + bcolors.ENDC

    TYPE_ERROR  = '{color}{obj}: expected {valid_types}, not {act_type}' + bcolors.ENDC

    VALID_BIN_VALUES = '{color}\'bins\' can only take values of 3, 4, 5, or 10' + bcolors.ENDC

    SORTING_FUNCS_BINS = '{color}\'bins\' and \'sorting_funcs\' cant both be \'None\'' + bcolors.ENDC

    BOTH_NONE = '{color}\'{var1}\' cannot be \'None\' if \'{var2}\' is used' + bcolors.ENDC

    MANY_COLS_TO_PRINT = '{color}Table {tab} has more than {lim} columns. Would you like to print all columns? [y/n]: ' + bcolors.ENDC

    NO_VALID_QTILE = '{color}No valid vars or qtile combination given.' + bcolors.ENDC
    
    STOCKS_NOT_SORTABLE = '{color}There are stocks that could not be sorted by {rank_col}. They will be removed before constructing portfolios.' + bcolors.ENDC
    
    DECOMPRESS_FAIL = '{color}Could not decompress file {file}.' + bcolors.ENDC
    
    TABLES_TO_DECOMPRESS = '{color}\'{obj}\' must be of type `list[str]` or `None`.'
    
    MUST_BE_DIRECTORY = '{color}\'{obj}\' must be a directory.' + bcolors.ENDC
    
    NOT_FOUND = '{color}\'{obj}\' does not exist.' + bcolors.ENDC
    
    PATH_FORMAT = '{color}\'{obj}\' must be castable to pathlib.Path.' + bcolors.ENDC
    
    DATE_FORMAT = '{color}\'{date}\' must be of type \'datetime.datetime\' or \'str\' in \'%Y-%m-%d\' format.' + bcolors.ENDC

    NO_DATA_FILE = '{color}There is no data file with the name {tab} in FILEStoDB/.' + bcolors.ENDC

    COMPRESS_FILES = '{color}It is best practice to compress the files read into the database.' + bcolors.ENDC

    FILES_TO_DB_SUCCESS = '{color}All files in \'DatabaseContents.Tables.FILES_TABLES\' have been added to the database.' + bcolors.ENDC

    FIRST_BUILD = '{color}This is the first time database {db} has been created. Please update the database structure to add data.' + bcolors.ENDC

    MISSING_WRDS_USERNAME = ('{color}Wharton Research Data Services (WRDS) username not given in <DatabaseContents.Tables>. Please update your WRDS username to download files from WRDS.'\
                            'If you do not want to download tables from WRDS set the attribute <DatabaseContents.Tables.WRDS_TABLES> to the empty list.') + bcolors.ENDC

    DATABASE_INITIALIZED = '{color}Database has been initialized! {time}s' + bcolors.ENDC

    DOWNLOAD_TABLES_CRASH = '{color}An error has occurred while downloading tables form WRDS. Normally this is a result of a lack of memory resources.' + bcolors.ENDC

    CREATE_TABLE_CRASH = '{color}The subprocess used to create table {tab} has failed.' + bcolors.ENDC

    EXTRA_TABLE_CRASH = '{color}The subprocess used to create table {tab} has failed.' + bcolors.ENDC

    NO_TABLES_CLASS_IN_DBC = '{color}\'Tables\' must be included in the \'DatabaseContents.py\' file.' + bcolors.ENDC

    REQUIRED_ATTRIBUTES_MISSING = '{color}Class \'{tab}\' is missing the required attributes {attr}.' + bcolors.ENDC

    ADDVARS_VARS_KWRDS = '{color}Keyword Arguments \'add_vars\' or \'sub_vars\' and \'vars\' cannot be used simultaneously' + bcolors.ENDC

    UPDATING_ALL_TABLES = '{color}Updating all of the tables in the local database. This process could take a long time. Are you sure you want to continue? [y/n]: ' + bcolors.ENDC

    MISSING_TABLE = '{color}The following tables are missing from the local database: {obj}. Querying WRDS to add them to the local database.' + bcolors.ENDC

    VAR_CANNOT_BE_QUERIED = '{color}Variables {obj} cannot be queried/used for filtering from {tab}. Check to make sure all variables are correct.' + bcolors.ENDC

    ABORT_INIT = '{color}Aborting database initialization' + bcolors.ENDC

    RAW_WRDS_ADDED = '{color}Raw WRDS files have been added to the local database.' + bcolors.ENDC

    TABLE_ADDED = '{color}Table added: {time}s' + bcolors.ENDC

    CLEANING_TABLE = '{color}Cleaning table {tab}...' + bcolors.ENDC

    TABLE_CLEANED = '{color}Table cleaned: {time}s' + bcolors.ENDC

    BUILDING_TABLE = '{color}Creating table {tab} using {file}...' + bcolors.ENDC

    DROPPING_TABLE = '{color}Dropping table {obj} from database' + bcolors.ENDC

    DELETE_DATABASE = '{color}Deleting database' + bcolors.ENDC

    CSV_TO_SQL_FAIL = '{color}Adding CSV to SQL database has failed' + bcolors.ENDC

    ABORT_OPERATION = '{color}{obj} operation aborted' + bcolors.ENDC

    COMPROMISE_DATABASE = ('{color}The operation that you are about to perform might compromise the local database.'\
                           'Operation of the <LocalDatabase> class might be affected. Do you wish to continue? [y/n]: ') + bcolors.ENDC

    UPDATING_OG_TABLE = '{color}You are updating a original data file used to make some derivative tables. Would you like to update all derivative tables? [y/n]: ' + bcolors.ENDC

    CONFIRM_DELETE = '{color}Are you sure you want to delete the following tables {obj}? [y/n]: ' + bcolors.ENDC

    MISSING_REQUIRED_SQL_COMPONENTS = '{color}The following required sql component is missing. {obj}' + bcolors.ENDC

    INVALID_COMPONENT_TYPE = '{color}Only objects of type \'int\', \'float\', \'str\', or \'list\' can be passed as a component. Check filtering for value {obj}' + bcolors.ENDC

    INVALID_SQL_CLEANING_OPERATION = '{color}The table {tab} has been given an invalid operation in \'DatabaseContents.Tables.SQL_CLEANING\'.' + bcolors.ENDC

    CSV_ADD_TABLE = '{color}Adding {tab} to SQL database {db}...' + bcolors.ENDC

    FINISHED_CSV_ADDING_CSV_TABLE = '{color}Finished {name}: {time}s' + bcolors.ENDC

    EXTRA_SCRIPT = '{color}Executing {path_to_execute}...' + bcolors.ENDC

    FILE_CONFLICT = '{color}There are two files in {path_to_folder} with table name {table} \'from DatabaseContents.Tables.FILES_TABLES\'. Please correct the conflict.' + bcolors.ENDC

    DOWNLOAD_SIC = '{color}Downloading SIC codes classifications from Ken French\'s website...' + bcolors.ENDC
    
    DOWNLOAD_SIC_CRASH = '{color}An error has occurred while downloading SIC code classifications from Ken French\'s website.' + bcolors.ENDC

    SIC_ADDED = '{color}SIC codes classifications have been added to the local database.' + bcolors.ENDC

    SQLITE_NULL_TABLE_CRASH = '{color}The sqlite_null_table.sh script has failed. Automatic type inference may be affected.' + bcolors.ENDC
    
    SQLITE_TYPE_INFERRING = '{color}Applying automatic type inference to table {tab}...' + bcolors.ENDC
    
    SQLITE_TYPE_INFER_CRASH = '{color}The sqlite_type_infer.sh script has failed. Automatic type inference may be affected.' + bcolors.ENDC

    TABLE_TYPE_INFERRED = '{color}Table {tab} has been type inferred: {time}s' + bcolors.ENDC