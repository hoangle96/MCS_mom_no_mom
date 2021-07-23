import numpy as np
import pandas as pd
from itertools import chain
import pickle
from variables import toys_dict, tasks, toys_list, toys_of_interest_dict
from visualization import draw_timeline_with_feature, save_png
from typing import List 
from read_data import get_consecutive_toys, remove_small_no_ops
from collections import defaultdict
if __name__ == '__main__':
    # load data
    with open('./data/interim/20210718_clean_data_for_feature_engineering.pickle', 'rb') as f:
        task_to_storing_dict = pickle.load(f)
    # toys_of_interest_dict = defaultdict()
    # toys_of_interest_dict['MPS'] = {'shape_sorter': 'shape_sorter_blocks'}
    # toys_of_interest_dict['NMS'] = {'shape_sorter': 'shape_sorter_blocks'}
    # toys_of_interest_dict['NMM'] = {'bucket':['food', 'balls'], 'grocery_cart':['food', 'balls']}
    # toys_of_interest_dict['MPM'] = {'bucket':['food', 'balls'], 'grocery_cart':['food', 'balls']}

    print(toys_of_interest_dict)
    subj_list = list(task_to_storing_dict['MPS'].keys())
    for subj in subj_list:
        df = pd.DataFrame()
        for task in tasks:
            # if subj == 11 and task == 'MPM':
                # print('here')
            # print(task_to_storing_dict[task][subj].columns)
            # print(task_to_storing_dict[task][subj],)
            # print(subj, task)
            # df_ = remove_small_no_ops(task_to_storing_dict[task][subj], small_no_ops_threshold = 3000)
            # print(df_)
            # df_ = get_consecutive_toys(df_, toys_of_interest_dict[task])
            df_ = task_to_storing_dict[task][subj]
            df = pd.concat([df, df_[['onset', 'offset', 'toy']]])
        df.to_csv("./data/interim/merged_actions/"+str(subj)+".csv", index = False)
        