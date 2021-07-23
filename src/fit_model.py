import pickle
import numpy as np
import pomegranate as pom
import sys
from visualization import draw_timeline_with_merged_states, save_png, draw_distribution

from variables import toys_dict, tasks, toys_list
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import check_random_state

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

from pathlib import Path
import os 
import importlib
from collections import OrderedDict

from merge import merge_segment_with_state_calculation_all as merge_state

def convert_to_int(list_to_convert):
    converted_list = []
    for i in list_to_convert:
        i = i.astype(int)
        converted_list.append(i)
    return converted_list

def convert_to_list_seqs(big_seq, len_array):
    big_seq_to_slice = big_seq.copy()
    list_of_seqs = []
    
    for k in len_array:
        list_of_seqs.append(big_seq_to_slice[:k])
        big_seq_to_slice = big_seq_to_slice[k:]
    return list_of_seqs

def save_csv(df, file_path, file_name):
    if not os.path.exists(file_path):
        file_path.mkdir(parents=True)
    save_path = file_path / file_name
    df.to_csv(save_path)

def discritize_with_sub(feature, threshold = 4):
    """
    Cap out the values at the higher end. Used for 'n_toys' and 'n_new_toys' features
    """
    return np.where(feature > threshold, threshold, feature)

def discritize_with_bins(feature, bins_):
    """
    Group values into different bins. Used for 'n_toy_switches" and 'fav_toy_ratio'
    """
    return np.digitize(feature, bins_, right = False)

def discritize_toy_iou(feature, discretizer):
    return discretizer.transform(feature).reshape((-1,))

def create_independent_dist(feature, seed):
    unique_val = np.unique(feature)
    init_dict = {}
    random_state = check_random_state(seed)
    init_prob = random_state.rand(len(unique_val),1)
    init_prob = init_prob/init_prob.sum()
    
    for idx, i in enumerate(unique_val):
        init_dict[int(i)] = init_prob[idx].item()
    return pom.DiscreteDistribution(init_dict)

def create_no_ops_state(feature):
    unique_val = np.unique(feature)
    init_dict = {}
    # print(unique_val)
    for idx, i in enumerate(unique_val):
        if idx == 0:
            init_dict[int(i)] = 1
        else:
            init_dict[int(i)] = 0
    return pom.DiscreteDistribution(init_dict)


def create_dist_for_states(n_states, feature_list, seed):
    distributions = []
    i = 0
    for s in range(n_states):
        if s == 0:
            dist_list = []
            for f in feature_list:
                dist = create_no_ops_state(f)
                dist_list.append(dist)
            distributions.append(pom.IndependentComponentsDistribution(dist_list))

        else:
            dist_list = []
            for f in feature_list:
                dist = create_independent_dist(f, i)
                i += 1
                dist_list.append(dist)
            distributions.append(pom.IndependentComponentsDistribution(dist_list))
    return distributions

def init_hmm(n_components, feature_list, seed):
    random_state_trans = check_random_state(seed**seed)
    transitions = random_state_trans.rand(n_components, n_components)
    transitions = transitions/transitions.sum()
    
    random_state_start = check_random_state(seed**2)
    starts = random_state_start.rand(n_components)
    starts = starts/starts.sum()
    distributions = create_dist_for_states(n_components, feature_list, seed)
    state_names = ["no_toys"] + [None]*(n_components-1)
    model = pom.HiddenMarkovModel.from_matrix(transitions, distributions, starts, state_names = state_names)   
    return model

if __name__ == "__main__":
    # load data

    interval_length = 2
    n_features = 4
    n_states = 6 
    with open('./data/interim/20210721_feature_engineering_'+str(interval_length)+'_min.pickle', 'rb') as f:
        feature_dict = pickle.load(f)

    with open('./data/interim/20210721_label_'+str(interval_length)+'_min.pickle', 'rb') as f:
        labels_dict = pickle.load(f)

    with open('./data/interim/20210721_clean_data_for_feature_engineering.pickle', 'rb') as f:
        task_to_storing_dict = pickle.load(f)

    with open('./data/interim/20210721_feature_engineering_time_arr_'+str(interval_length)+'_min.pickle', 'rb') as f:
        time_arr_dict = pickle.load(f)
    
    new_toy_threshold = 2

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
    
    # Discretize feature 

    toy_switch_bins = [0, 4, 8, 12]
    discretized_toy_switch_rate = np.digitize(input_list[:,0], toy_switch_bins, right = False)

    discretized_n_toys = np.where(input_list[:,1] > 4, 4, input_list[:,1])  

    discretized_n_new_toys = np.where(input_list[:,2] > 4, 4, input_list[:,2])
    
    fav_toy_bin = [0, .2, .4, .6, .8]
    fav_toy_rate_discretized = np.digitize(input_list[:,3].copy(), fav_toy_bin, right = False)
    
    discretized_input_list = np.hstack((discretized_toy_switch_rate.reshape((-1,1)),\
                                    discretized_n_toys.reshape((-1,1)),\
                                    discretized_n_new_toys.reshape((-1,1)),\
                                    fav_toy_rate_discretized.reshape((-1,1))))

    list_seq = convert_to_list_seqs(discretized_input_list, len_list)
    list_seq = convert_to_int(list_seq)

    model = init_hmm(n_states, discretized_input_list.T, seed = 1)
    model.bake()
    # freeze the no_toys distribution so that its parameters are not updated. 
    # "no_toys" state params are set so that all of the lowest bins = 0
    for s in model.states:
        if s.name == "no_toys":
            for p in s.distribution.parameters[0]:
                p.frozen = True
    model.fit(list_seq, labels = all_labels)

    model_file_path = Path('./models/hmm/20210721/'+str(n_states)+'_states_'+str(interval_length)+'_min.pickle')
    with open(model_file_path, 'wb+') as f:
        pickle.dump(model, f)


    i = 0
    input_dict = {}
    for task in tasks:
        if task not in input_dict.keys():
            input_dict[task] = {}

        for subj, shifted_df_dict in feature_dict[task].items():
            if subj not in input_dict[task].keys():
                input_dict[task][subj] = {}


            for shift_time, feature_vector in shifted_df_dict.items():
                input_dict[task][subj][shift_time] = list_seq[i]
                i += 1

    total_log_prob = 0
    log_prob_list = []
    pred_dict = {}
    proba_dict = {}
    all_proba_dict = {}

    pred_by_task = {}
    input_by_task = {}

    for task in tasks:
        if task not in pred_dict.keys():
            pred_dict[task] = {}
            proba_dict[task] = {}
            all_proba_dict[task] = {}
            pred_by_task[task] = []
            input_by_task[task] = []

        for subj, shifted_dict in input_dict[task].items():
            if subj not in pred_dict[task].keys():
                pred_dict[task][subj] = {}
                proba_dict[task][subj] = {}
                all_proba_dict[task][subj] = {}

            for shift_time, feature_vector in shifted_dict.items():
                label= model.predict(feature_vector)
                pred_dict[task][subj][shift_time] = label
                pred_by_task[task].extend(label)
                input_by_task[task].extend(feature_vector)

                # if 4 in label:
                    # print(feature_vector, label)
                proba_dict[task][subj][shift_time] = np.amax(model.predict_proba(feature_vector), axis = 1)
                log_prob = model.log_probability(feature_vector)
                all_proba_dict[task][subj][shift_time] = model.predict_proba(feature_vector)
                
                log_prob_list.append(log_prob)
    print(np.mean(log_prob_list))

    with open('./data/interim/20210721_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'wb+') as f:
        pickle.dump(pred_dict, f)
    # subj_list = list(task_to_storing_dict['MPS'].keys())
    # shift_time_list = np.arange(0,interval_length,0.25)

    # merged_pred_dict_all = {}
    # merged_proba_dict_all = {}
    # time_subj_dict_all = {}
    # for task in tasks:
    #     print(task)
    #     merged_df_dict = task_to_storing_dict[task]
    #     time_arr_shift_dict = time_arr_dict[task]
    #     pred_subj_dict = pred_dict[task]
    #     prob_subj_dict = all_proba_dict[task]

    #     merged_pred_dict_all_task_specific, merged_proba_dict_all_task_specific, time_subj_dict_all_task_specific = merge_state(subj_list, shift_time_list, merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size = interval_length, n_states = 5, shift_interval = 60000*.25)

    #     merged_pred_dict_all[task] = merged_pred_dict_all_task_specific
    #     merged_proba_dict_all[task] = merged_proba_dict_all_task_specific
    #     time_subj_dict_all[task] = time_subj_dict_all_task_specific
    
    # with open('./data/interim/20210721_5_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    #     pickle.dump(merged_pred_dict_all, f)

    # with open('./data/interim/20210721_5_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    #     pickle.dump(merged_proba_dict_all, f)
    
    # with open('./data/interim/20210721_5_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    #     pickle.dump(time_subj_dict_all, f)
    

    