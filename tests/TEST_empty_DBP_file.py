import os
import sys
import time
import numpy
import pandas
import pathlib
import datetime
import functools
import knockknock
import matplotlib.pyplot as plt

# add source directory to path
sys.path.insert(0, '../src/FinToolsAP/')

import LaTeXBuilder
import LocalDatabase
import PortfolioSorts
import UtilityFunctions

# set printing options
import shutil
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', shutil.get_terminal_size()[0])
pandas.set_option('display.float_format', lambda x: '%.3f' % x)

# directory for loacl wrds database
db_path = pathlib.Path('/home/andrewperry/Nextcloud/Research/CreditModel')

def main():
    
    DB = LocalDatabase.LocalDatabase(db_path, 'CreditModelDB')
    
    
    
    
    
WEBHOOK_URL = 'https://discord.com/api/webhooks/1274502660506128467/TsUIvv94xmVubcHDFG9CRB2MgtwltiogNcd2tG7xhxuipC-sglXMx1nxibtuRQarsCt-'
@knockknock.discord_sender(webhook_url = WEBHOOK_URL)
def TEST_empty_DBP_file(): # change to name of file
    main()

if __name__ == '__main__':
    if(os.getlogin() == 'andrewperry' and True):
        TEST_empty_DBP_file() # change to name of file
    else:
        main()