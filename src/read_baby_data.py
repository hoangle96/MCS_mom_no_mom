import pandas as pd
import numpy as np
import glob
from feature_engineering import shift_signal_toy_count
from visualization import save_png, draw_plain_timeline
from pathlib import Path
from variables import toys_dict, tasks, toys_list, toys_of_interest_dict
import pickle 
import itertools
from datetime import datetime 

if __name__ == '__main__':
    # pre-defined vars
    CHECKING = False # if True will export the CSV and draw out the interaction timeline

    mps_dict = {}
    mpm_dict = {}
    nms_dict = {}
    nmm_dict = {}

    infant_info = {'walking_exp':{}, 'crawling_exp':{}}
    
    toy_names = {str(v): k for k, v in toys_dict.items()}

    subj_dict = {}

    data_path = './data/raw/id/*.csv'
    for file_name in glob.glob(data_path):
        print(file_name)
        subj = int(file_name.split('/')[-1].split('.')[0])
        # if subj == 3:

        

        print(subj)
        # if subj != 39 and subj != 4 or (subj != 43 and task != 'NMM'):
        # if subj == 3:
        #     print('here')
        df = pd.read_csv(file_name)
        # print(df)
        # print(df[['id.tdate', 'id.walkdate', 'id.cruisedate', 'id.hkdate']])

        # print('testdate', df['id.tdate'].values)
        if len(df['id.tdate'].values[0].split('/')[-1]) == 2:
            test_date = datetime.strptime(df['id.tdate'].values[0], "%m/%d/%y")
        else:
            test_date = datetime.strptime(df['id.tdate'].values[0], "%m/%d/%Y")

        # print('walkdate', df['id.walkdate'].values)
        if df['id.walkdate'].values[0] != '.':
            if len(df['id.walkdate'].values[0].split('/')[-1]) == 2:
                walking_date = datetime.strptime(df['id.walkdate'].values[0], "%m/%d/%y")
            else:
                walking_date = datetime.strptime(df['id.walkdate'].values[0], "%m/%d/%Y")
            infant_info['walking_exp'][subj] = abs((test_date - walking_date).days)
            

        # cruise_date = datetime.strptime(df['id.cruisedate'].values[0], "%m/%d/%y")
        # print('crawldate', df['id.hkdate'].values)
        if df['id.hkdate'].values[0] != '.':
            if len(df['id.hkdate'].values[0].split('/')[-1]) == 2:
                crawling_date = datetime.strptime(df['id.hkdate'].values[0], "%m/%d/%y")
            else:
                crawling_date = datetime.strptime(df['id.hkdate'].values[0], "%m/%d/%Y")
            infant_info['crawling_exp'][subj] = abs((test_date - crawling_date).days)




        # infant_info['cruising_exp'][subj] = abs((test_date - cruise_date).days)




    for exp_name, exp in infant_info.items():
        print(len(exp))
        print()
        print(exp_name, exp)

    with open('./data/interim/20210818_baby_info.pickle', 'wb+') as f:
        pickle.dump(infant_info, f)