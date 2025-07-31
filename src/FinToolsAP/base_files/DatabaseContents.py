# every table must have 'TABLE', 'DEFAULT_VARS', 'VARS_DATA_TYPE', 'DEFAULT_ID', 'DEFAULT_DATE'

class Tables:

    # The order indicates the order by which data is added to the database

    # 1. ExtraScripts/ 
    # list of tables added using ExtraScripts/
    EXTRA_TABLES = []

    # 2. FILEStoDB/
    # list of tables to read in from FILEStoDB/
    FILES_TABLES = []

    # 3. WRDS
    # list of tables to download from WRDS
    WRDS_USERNAME = None
    WRDS_TABLES = [] 

    # 4. Cleaning
    # additional cleaning operations to be applied to data
    # format {'tablename': {'operation': ['columns']}}
    SQL_CLEANING = {}

    # 5. CREATED TABLES
    # list of created tables using scripts in CreateTables/
    CREATED_TABLES = []

# Many different ways to organize your data
    
class YourCustomTableNmaeExample:
    # fille this out with your own table information

    TABLE = None
    DEFAULT_VARS = []
    VARS_DATA_TYPE = {}
    DEFAULT_ID = None
    DEFAULT_DATE = None

class DataVendorExample:

    class VendorTable1Example:
        TABLE = None
        DEFAULT_VARS = []
        VARS_DATA_TYPE = {}
        DEFAULT_ID = None
        DEFAULT_DATE = None


    class VendorTable2Example:
        TABLE = None
        DEFAULT_VARS = []
        VARS_DATA_TYPE = {}
        DEFAULT_ID = None
        DEFAULT_DATE = None

class Project1Example:

    class Table1Example:
        TABLE = None
        DEFAULT_VARS = []
        VARS_DATA_TYPE = {}
        DEFAULT_ID = None
        DEFAULT_DATE = None

    class Table2Example:
        TABLE = None
        DEFAULT_VARS = []
        VARS_DATA_TYPE = {}
        DEFAULT_ID = None
        DEFAULT_DATE = None

class Project2Example:

    class Table1Example:
        TABLE = None
        DEFAULT_VARS = []
        VARS_DATA_TYPE = {}
        DEFAULT_ID = None
        DEFAULT_DATE = None

    class Table2Example:
        TABLE = None
        DEFAULT_VARS = []
        VARS_DATA_TYPE = {}
        DEFAULT_ID = None
        DEFAULT_DATE = None

