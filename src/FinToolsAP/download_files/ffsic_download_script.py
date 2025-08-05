# DO NOT MODIFY

import sys
import pathlib
import sqlalchemy

PATH_TO_DB = sys.argv[1]

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

# standard imports
import io
import re
import pandas
import zipfile
import requests

def is_valid_sic_range(s: str) -> bool:
    return bool(re.compile(r'^\d{4}-\d{4}$').match(s))

# fama-french industy classification levels
ff_levels = [5, 10, 12, 17, 30, 38, 48, 49]

for ffcode in ff_levels:

    print(f'{_config.bcolors.OKCYAN}Downloading classification level: {ffcode}...' + _config.bcolors.ENDC)

    ff_url = f'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes{ffcode}.zip'
    resp = requests.get(ff_url, timeout=30)
    resp.raise_for_status()  
    
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw = zf.read(txt_name).decode("latin1")   # Ken French files are ANSI/Latin-1
        
    res = []
    curr_id = None
    curr_abbrev = None
    curr_group = None
    for line in raw.splitlines():

        line_test = line.strip()
        ln = [l.strip() for l in line_test.split(' ', 1)]
        if ln[0] == '':
            continue
        elif is_valid_sic_range(ln[0]):
            # extract SIC range and get Description
            sic_start, sic_end = map(int, ln[0].split('-'))
            desc = ln[1] if len(ln) > 1 else ''
            res.append({
                "industry_id": curr_id,
                "abbrev": curr_abbrev,
                "industry_group": curr_group,
                "description": desc,
                "sic_start": sic_start,
                "sic_end": sic_end
            })
        else:
            # header line
            header = [l.strip() for l in line_test.split(' ', 2)]
            curr_id = header[0]
            curr_abbrev = header[1]
            curr_group = header[2]
            continue
        
    df = pandas.DataFrame(res)
    
    df.to_sql(f'FFSIC_{ffcode}', con = sql_engine, if_exists = 'replace', index = False)

place_holder = pandas.DataFrame({'place': ['holder']})
place_holder.to_sql('FFSIC', con = sql_engine, if_exists = 'replace', index = False)