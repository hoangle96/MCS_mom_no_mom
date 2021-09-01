import pickle
import numpy as np
import pomegranate as pom
# import sys
# sys.path.append('../src')
from visualization import draw_plain_timeline_with_feature_discretization
from hmm_model import convert_to_int, convert_to_list_seqs, save_csv, init_hmm

from variables import toys_dict, tasks, toys_list
import matplotlib.pyplot as plt

from merge import merge_segment_with_state_calculation_all, merge_toy_pred
import pandas as pd

from pathlib import Path
import importlib
from collections import OrderedDict

if __name__ == '__main__':
    feature_set = 'n_new_toy_ratio'
    for no_ops_threshold in [10, 7, 5]:
        with open('./data/interim/20210805_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
            task_to_storing_dict = pickle.load(f)
        subj_list = task_to_storing_dict['MPS'].keys()

        for interval_length in [1.5, 2, 1]:
            with open("./data/interim/20210815_"+str(no_ops_threshold)+"_no_ops_threshold_feature_dict_with_"+feature_set+"_"+str(interval_length)+"_min.pickle", 'rb') as f:
                feature_dict = pickle.load(f)

            with open("./data/interim/20210815_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
                time_arr_dict = pickle.load(f)

            for task in tasks:
                for subj in subj_list:
                    df = task_to_storing_dict[task][subj]
                    feature = feature_dict[task][subj][0]
                    time_arr_list = time_arr_dict[task][subj][0]
                    df_ = pd.DataFrame()
                    for df__ in df:
                        df_ = pd.concat([df_, df__])
                    fig_name = './figures/hmm/state_distribution_20210815_30s/'+str(feature_set)+'/no_ops_threshold_' + str(no_ops_threshold)\
                        +'/window_size_' +str(interval_length)+ '/feature_engineering/' +task+'/'+str(subj)+".png"
                    feature = feature.reshape((-1, 4))
                    draw_plain_timeline_with_feature_discretization(
                        subj, df_, time_arr_list, features=feature, gap_size=interval_length, fav_toy_list=None, fig_name=fig_name)

