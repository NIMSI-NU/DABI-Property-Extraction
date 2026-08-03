import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from pathlib import Path
import io
import re
import DABI_manipulator as dm

# Change the following variables to determine where the different files and paths are 
instrument_root = r'D:\OneDrive\Northwestern University\DARPA RADICAL - RADICAL Documents\FA 1\DABI phase 2\High Pressure DABI\Code\RT DABI Software\Test Files'
instrument_file = r'RTTest_B3_17_55_01.CSV'
dic_root = r'D:\OneDrive\Northwestern University\DARPA RADICAL - RADICAL Documents\FA 1\DABI phase 2\High Pressure DABI\Testing\(250602)\B3\left'

instrument_type = r'lt-dabi'

timestamp_title_dict = {'lt-dabi': 'TimeStamp',
                        'ht-dabi': 'Unix Time'}

instrument_df = dm.load_data(instrument_root, instrument_file, instrument_type)
print('loaded instrument data')
dic_df = dm.load_data(dic_root, r'r', 'dic')
print('loaded dic data')
dic_df.to_csv(os.path.join(dic_root, "timestamp_array.csv"))

if instrument_type == r'ht-dabi':
    instrument_df_unix = dm.convert_to_unix(instrument_df, '', '%m/%d/%Y %H:%M:%S.%f', 0)
    print('converted to unix')
elif instrument_type == r'lt-dabi':
    instrument_df_unix = instrument_df
    print('room temperature code does not require unix conversion')

aligned_df = dm.align_data(dic_df, instrument_df_unix, instrument_time_header=timestamp_title_dict[instrument_type])
print('done with alignment, total ' + str(len(dic_df)) + ' displacement fields')

aligned_df.to_csv(os.path.join(instrument_root, Path(instrument_file).stem + "_aligned.csv"))
print('saved to disk')