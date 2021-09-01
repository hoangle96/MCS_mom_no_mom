#%%
import pickle
import numpy as np
import pomegranate as pom
import sys
# sys.path.append('../src')
from visualization import draw_mean_state_locotion_across_conditions, draw_mean_state_locotion_across_conditions_separate_mean_std

from variables import toys_dict, tasks, toys_list, condition_name
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
import seaborn as sns

# %%
def rank_state(model):
    # label_name_list = []
    # n_toy_list = []
    state_label_n_toy_dict = {}
    for idx, s in enumerate(model.states):
        if not s.distribution is None:
            if s.name == "no_toys":
                state_label_n_toy_dict[idx] = 0
            else: 
                # print(s.distribution.parameters[0][1].parameters[0])
                state_label_n_toy_dict[idx] = np.dot(np.array(list(s.distribution.parameters[0][1].parameters[0].values())),np.array(list(s.distribution.parameters[0][1].parameters[0].keys())).T) 
    # print(state_label_n_toy_dict)
    ranked_dict = {k: v for k, v in sorted(state_label_n_toy_dict.items(), key=lambda item: item[1])}
    return {v: str(k) for k, v in enumerate(ranked_dict.keys())}


for feature_set in ['n_new_toy_ratio']:
    for no_ops_time in [10, 5,7]:
        for interval_length in [1.5, 2, 1]:
            for n_states in range(5, 7):
                with open('../data/interim/20210816_30s_offset_new_merge_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_locomotion_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    merged_pred_w_locomotion = pickle.load(f)
                

                model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
                model_file_path = Path('../models/hmm/20210815_30s_offset/'+feature_set)/model_file_name
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)
                state_name_dict = rank_state(model)

                movement_time_by_each_task = {}
                steps_by_each_task = {}
                movement_time_by_each_task_for_std = {}
                steps_by_each_task_for_std = {}
                movement_time_by_each_task_mean_each_infant = {}
                steps_by_each_task_mean_each_infant = {}

                

                # movement_time_by_each_state = {}
                # steps_by_each_state = {}

                for task in tasks:
                    if task not in movement_time_by_each_task.keys():

                        movement_time_by_each_task[task] = {}
                        steps_by_each_task[task] = {}

                        steps_by_each_task_for_std[task]= {}
                        movement_time_by_each_task_for_std[task] = {}

                        movement_time_by_each_task_mean_each_infant[task] = {}
                        steps_by_each_task_mean_each_infant[task] = {}



                        for state in range(n_states):
                            movement_time_by_each_task[task][state] = []
                            steps_by_each_task[task][state] = []

                            movement_time_by_each_task_mean_each_infant[task][state] = []
                            steps_by_each_task_mean_each_infant[task][state] = []

                            # steps_by_each_task_for_std[task][state] = 
                            # movement_time_by_each_task_for_std[task][state] = []


                            # movement_time_by_each_state[state] = []
                            # steps_by_each_state[state] = []


                    for subj, df in merged_pred_w_locomotion[task].items():
                        # movement_time_by_state[task].append()
                        # print(state_name_dict)
                        # print(df)
                        df['pred'] = df['pred'].replace(state_name_dict)
                        for state in range(n_states):
                            # print(state)
                            df_ = df.loc[df.loc[:,'pred'] == str(state),:]
                            

                            if len(df_) > 0:
                                steps = df_['steps'].to_numpy()*4
                                movement_time = df_['movement_time'].to_numpy()/15000
                                steps_mean = np.mean(df_['steps'].to_numpy()*4)
                                movement_time_mean = np.mean(df_['movement_time'].to_numpy()/15000)

                                
                            else:
                                steps = []
                                movement_time = []
                                steps_mean = 0
                                movement_time_mean = 0
                                steps_by_each_task_for_std[task][state] = None
                                movement_time_by_each_task_for_std[task][state] = None


                                

                            # print(df.loc[df.loc[:,'pred'] == str(state), 'steps'].to_numpy()*4)
                            # steps = np.nanmean(df.loc[df.loc[:,'pred'] == str(state), 'steps'].to_numpy()*4)
                            
                            # movement_time = np.nanmean(df.loc[df.loc[:,'pred'] == str(state), 'movement_time'].to_numpy()/15000)

                            steps_by_each_task[task][state].extend(steps)
                            movement_time_by_each_task[task][state].extend(movement_time)

                            movement_time_by_each_task_mean_each_infant[task][state].append(movement_time_mean)
                            steps_by_each_task_mean_each_infant[task][state].append(steps_mean)


                            # steps_by_each_task[task][state].append(steps_mean)
                            # movement_time_by_each_task[task][state].append(movement_time_mean)

                            # movement_time_by_each_state[state].extend(steps)
                            # steps_by_each_state[state].extend(movement_time)
                    for state in range(n_states):

                        steps_by_each_task_for_std[task][state] = np.sqrt(np.mean(np.abs(movement_time_by_each_task_mean_each_infant[task][state]-np.mean(steps_by_each_task[task][state]))**2))
                        movement_time_by_each_task_for_std[task][state] = np.sqrt(np.mean(np.abs(movement_time_by_each_task_mean_each_infant[task][state]-np.mean(movement_time_by_each_task[task][state]))**2))




                # fig_path = './figures/hmm/state_distribution_20210805/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/step_by_state_2.png"

                # draw_mean_state_locotion_across_conditions(data_dict=steps_by_each_task,\
                #                                             task_list = ["MPM", "NMM", "MPS", "NMS"],\
                #                                             condition_name = condition_name,\
                #                                             n_states = n_states, \
                #                                             ylabel = 'avg # steps/min',\
                #                                             title = "Avg number of steps in each state for each condition, " +str(no_ops_time) + "s threshold, window size " +str(interval_length),\
                #                                             figname = fig_path)

                fig_path = '../figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/step_by_state_3.png"

                draw_mean_state_locotion_across_conditions_separate_mean_std(mean_dict =steps_by_each_task,\
                                                            std_dict = steps_by_each_task_for_std,\
                                                            task_list = ["MPM", "NMM", "MPS", "NMS"],\
                                                            condition_name = condition_name,\
                                                            n_states = n_states, \
                                                            ylabel = 'avg # steps/min',\
                                                            title = "Avg number of steps in each state for each condition,\n" +str(no_ops_time) + "s threshold, window size " +str(interval_length),\
                                                            figname = fig_path)

                # fig_path = './figures/hmm/state_distribution_20210805/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/loco_time_by_state_2.png"

                # draw_mean_state_locotion_across_conditions(data_dict=movement_time_by_each_task,\
                #                                             task_list = ["MPM", "NMM", "MPS", "NMS"],\
                #                                             condition_name = condition_name,\
                #                                             n_states = n_states, \
                #                                             ylabel = "% time in state",\
                #                                             title = "Pct. of session in motion in each state for each condition, " +str(no_ops_time) + "s threshold, window size " +str(interval_length),\
                #                                             figname = fig_path)
                
                fig_path = '../figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/loco_time_by_state_3.png"

                draw_mean_state_locotion_across_conditions_separate_mean_std(mean_dict = movement_time_by_each_task,\
                                                            std_dict = movement_time_by_each_task_for_std,\
                                                            task_list = ["MPM", "NMM", "MPS", "NMS"],\
                                                            condition_name = condition_name,\
                                                            n_states = n_states, \
                                                            ylabel = '% time in state',\
                                                            title = "Pct. of session in motion in each state for each condition,\n" +str(no_ops_time) + "s threshold, window size " +str(interval_length),\
                                                            figname = fig_path)

# %%
