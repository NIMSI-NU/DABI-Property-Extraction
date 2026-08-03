import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from pathlib import Path
import io
import re
import cv2

def load_data(root_path, filename, instrument):
    r'''Loads the csv data into a pandas data structure

    Args:
        root_path (str): folder containing file, without \ at the end of argument 
            Example: 'C:\Users\default_user\data_folder'
        filename (str): file with data and headers associated. Must be csv. Not used for DIC data
        instrument (str): Must be 'dic', 'lt-dabi', or 'ht-dabi' to not throw exception

    Returns:
        DataFrame: DataFrame that contains the csv data with associated headers
    '''
    if instrument.lower() == r'lt-dabi':
        with open(os.path.join(root_path, filename)) as table:
            buffer = io.StringIO('\n'.join(re.sub(r"[\r\t\s]*", "", line) for line in table))
            df = pd.read_table(buffer, low_memory = False, usecols=range(3), header=1, delimiter=',')
        return df
    
    if instrument.lower() == r'dic':
        
        # All files in the directory
        all_files = os.listdir(root_path)
        
        # List of all image files only
        imgnames = []
        for file_itr in all_files:
            if file_itr.endswith(".tiff"):
                imgnames.append(file_itr)
                
        num_imgs = len(imgnames)
        
        imagterations = np.zeros([num_imgs])
        unix_timestamps = np.zeros([num_imgs])
        
        # Iterate through image names
        for itr in range(num_imgs):
            substrings_itr = imgnames[itr].split("_")
            
            # Unix timestamp substring
            unix_timestamps[itr] = np.int64(substrings_itr[-2])
            
            # Iteration substring
            imagterations[itr] = np.int64(substrings_itr[-3])
            
        # sort array
        imagterations_sort = np.argsort(imagterations)
        timestamps_sort = np.argsort(unix_timestamps)
        
        # these should be the same vector
        if (imagterations_sort == timestamps_sort).all:
            pass
        else:
            raise Exception("Warning! Image timestamps out of order!")
        
        imagterations_sorted = imagterations[imagterations_sort]
        timestamps_sorted = unix_timestamps[timestamps_sort]
        timestamps_sorted_seconds = np.float64(np.float64(timestamps_sorted)*1.0e-3)
        
        debugfig, debugax = plt.subplots()
        debugax.plot(imagterations_sorted, timestamps_sorted)
        debugfig.savefig(os.path.join(root_path, "key.svg"))
        
        output_db = pd.DataFrame({'frame': imagterations_sorted, 'Unix Time': timestamps_sorted_seconds})
        return output_db
        
    elif instrument.lower() == r'ht-dabi':
        return pd.read_csv(os.path.join(root_path, filename), skiprows=1, low_memory = False)
    else:
        raise Exception("\n\nWrong instrument name used.\n" + 
                        f"'{instrument.lower()}' is not an allowed instrument type.\n" +
                        "Please use one of the following: 'dic', 'lt-dabi', or 'ht-dabi'")

def convert_to_unix(df, header, date_string, timezone = 0):
    r'''Converts a string time to unix
    
    Args:
        df (DataFrame): The dataframe with a string as the time
        header (str): The DataFrame header that the time column is labeled
        date_string (str): Format of the data, using datetime notation
        timezone (int): Optional, assumes UTC. Give int for timezone difference from UTC (timezone = 5 for CDT, timezone = 6 for CST)

    Returns:
        DataFrame: Returns the same dataframe structure, but with unix time as a float    
    '''
    unix_time = np.zeros(len(df[header])) #Zero array to start
    for i in range(len(df[header])):
        datetime_obj_local = datetime.strptime(df[header][i], date_string) #Strip the time based on formatting
        datetime_obj_utc = datetime_obj_local + timedelta(hours=timezone) #Set as datetime object with offset based on timezone
        unix_time[i] = datetime_obj_utc.timestamp() #.timestep() determines the seconds since Epoch 
    
    df_new = df.copy()
    df_new.loc[:,header] = unix_time
    return df_new

def align_data(dic_frame_time, instrument_data,  dic_time_header = 'Unix Time', dic_frame_header = 'frame', instrument_time_header = 'time'):
    r'''Aligns data from the insturment and aligns it to DIC

    Args:
        dic_frame_time (DataFrame): DataFrame that has two columns, the unix time and the frame number associated with that time
        instrument_data (DataFrame): DataFrame that contains the instrument data to be aligned
        dic_time_header (str): Header name for the time column in the dic file
        dic_frame_header (str): Header name for the frame label column, to be included in the final output dataframe
        instrument_time_header(str): Header for the time column in the instrument data

    Returns:
        DataFrame: The aligned data
    '''
    
    start_time = np.max([instrument_data[instrument_time_header][0], dic_frame_time[dic_time_header][0]])               # Determines the start time of the region with both DIC and DABI data
    end_time = np.min([instrument_data[instrument_time_header].iloc[-1], dic_frame_time[dic_time_header].iloc[-1]])     # Determines the end time of the region with both the DIC and DABI data
    
    start_ind = (dic_frame_time[dic_time_header] - start_time).abs().idxmin() #Find the start index of the DIC data
    end_ind = (dic_frame_time[dic_time_header] - end_time).abs().idxmin() #Find the end index of the DIC data
    
    #For each frame, find the instument line that corresponds to each DIC image
    data_ind = np.array([(instrument_data[instrument_time_header] - dic_frame_time[dic_time_header].iloc[start_ind + i]).abs().idxmin() for i in range(end_ind - start_ind + 1)])
    df_aligned = instrument_data.loc[data_ind].copy() #Copy only those indices to a new dataframe
    df_aligned.reset_index(drop = True, inplace = True) #Drop the old indices that are stored in the original dataframe (causes issues later)
    df_aligned[instrument_time_header] = dic_frame_time[dic_time_header] #Replace the instrument time with the DIC time
    df_aligned.insert(1, dic_frame_header, dic_frame_time[dic_frame_header]) #List the filename as the second column of the dataframe for each data point

    return df_aligned