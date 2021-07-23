import numpy as np
from sklearn.model_selection import KFold
import pickle
import os 
import pomegranate as pom
import sys
from sklearn.utils import check_random_state
from variables import toys_dict, tasks, toys_list
from hmm_model import kfold_each_window_size


            
if __name__ == "__main__":
    n_features = 4

    llh_score = {}
    n_bin_toy_switch_dict = {1:  [0, 4, 8, 12], 1.5:  [0, 5, 10, 15], 2:  [0, 4, 8, 12]}

    for interval_length in [1, 1.5, 2]:
        print("window_size " + str(interval_length))
        with open('./data/interim/20210721_feature_engineering_'+str(interval_length)+'_min.pickle', 'rb') as f:
            feature_dict = pickle.load(f)

        with open('./data/interim/20210721_label_'+str(interval_length)+'_min.pickle', 'rb') as f:
            labels_dict = pickle.load(f)

        shift_time_list = np.arange(0, interval_length, .25)

        len_list = []

        input_list = np.empty((0, n_features))
        input_list_ = np.empty((0, n_features))

        for task in tasks:
            for subj, shifted_df_dict in feature_dict[task].items():
                for shift_time, feature_vector in shifted_df_dict.items():
                    # print(feature_vector)
                    input_list = np.vstack((input_list, feature_vector))
                    input_list_ = np.concatenate((input_list_, feature_vector))
                    len_list.append(len(feature_vector))
        
        all_labels = []
        for task in tasks:
            for subj, shifted_sequence in labels_dict[task].items():
                for shift_time, label in shifted_sequence.items(): 
                    all_labels.append(label)

        list_of_feature = []
        for task in tasks:
            for subj, shifted_df_dict in feature_dict[task].items():
                for shift_time, feature_vector in shifted_df_dict.items():
                    list_of_feature.append(feature_vector)
        
        llh_score[interval_length] = kfold_each_window_size(list_of_feature, all_labels, n_bin_toy_switch_dict[interval_length], max_n_states = 7)
    
    with open('./data/interim/20210721_cross_val_new.pickle', 'wb+') as f:
        pickle.dump(llh_score, f)