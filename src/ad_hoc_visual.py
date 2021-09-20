#%%
import numpy as np 
import pandas as pd
from variables import tasks, condition_name, state_color_dict, stationary_toys_list, mobile_toys_list, state_color_dict_shades
import pickle 
from merge import merge_segment_with_state_calculation_all
from visualization import draw_toy_state, draw_distribution, draw_timeline_with_merged_states, draw_state_distribution, draw_toy_state_with_std, draw_infant_each_min_matplotlib, draw_mean_state_locotion_across_conditions, draw_timeline_with_prob_to_check, draw_mean_state_locotion_across_conditions_separate_mean_std
import os
from pathlib import Path 

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
                state_label_n_toy_dict[idx] = np.dot(np.array(list(s.distribution.parameters[0][1].parameters[0].values())),np.array(list(s.distribution.parameters[0][1].parameters[0].keys())).T)+\
                                              np.dot(np.array(list(s.distribution.parameters[0][2].parameters[0].values())),np.array(list(s.distribution.parameters[0][2].parameters[0].keys())).T)
                
    # print(state_label_n_toy_dict)
    ranked_dict = {k: v for k, v in sorted(state_label_n_toy_dict.items(), key=lambda item: item[1])}
    return {v: str(k) for k, v in enumerate(ranked_dict.keys())}

# %%
shift = .5
mobile_toys_list.append('no_toy')
stationary_toys_list.append('no_toy')
feature_set = 'n_new_toy_ratio'
no_ops_time = 10
interval_length = 1.5
n_states = 5

with open('../data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    pred_dict = pickle.load(f)
model_file_name = "model_20210907_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

if feature_set == 'n_new_toy_ratio' or feature_set == 'n_new_toy_ratio_and_fav_toy_till_now':
    x_ticks_dict = {0: ["[0, 4)", '[4, 8)', '[8, 12)', '[12+'], 1: ['0', '1', '2', '3', '4+'], 2: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
    feature_names = ["# toys switches", "# toys", "# new toys ratio", 'fav toy ratio']
    feature_values = {0: range(1,5), 1: range(5), 2: range(1, 6), 3: range(1,6)}
elif feature_set == 'fav_toy_till_now':
    x_ticks_dict = {0: ["[0, 4)", '[4, 8)', '[8, 12)', '[12+'], 1: ['0', '1', '2', '3', '4+'], 2: ['0', '1', '2', '3', '4+'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
    feature_names = ["# toys switches", "# toys", "# new toys", 'fav toy ratio']
    feature_values = {0: range(1,5), 1: range(5), 2: range(5), 3: range(1,6)}
elif feature_set == 'new_toy_play_time_ratio':
    x_ticks_dict = {0: ["[0, 4)", '[4, 8)', '[8, 12)', '[12+'], 1: ['0', '1', '2', '3', '4+'], 2: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
    feature_names = ["# toys switches", "# toys", "new toys play time ratio", 'fav toy ratio']
    feature_values = {0: range(1,5), 1: range(5), 2: range(1, 6), 3: range(1,6)}

n_features = 4
flatten_pred = []
flatten_pred_dict = {}
flatten_pred_dict_cg = {"without_cg":[], 'with_cg':[]}
flatten_pred_dict_toy_set = {'mobile':[], 'fine':[]}

Path('../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/").mkdir(parents=True, exist_ok=True)
for task in tasks:
    flatten_pred_dict[task] = []
    if task in ["MPM", "NMM"]:
        toy_task = 'mobile'
    else:
        toy_task = 'fine'
    
    if task in ['MPS', "MPM"]:
        mom_task = 'with_cg'
    else:
        mom_task = 'without_cg'

    task_specific_pred_dict = pred_dict[task]
    for subj, subj_dict in task_specific_pred_dict.items():
        for shift_time, pred in subj_dict.items():
            flatten_pred.extend(pred)
            flatten_pred_dict[task].extend(pred)
            flatten_pred_dict_cg[mom_task].extend(pred)
            flatten_pred_dict_toy_set[toy_task].extend(pred)


fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_mom.png"
draw_state_distribution(flatten_pred_dict_cg['with_cg'], n_states, state_name_dict, "With caregiver, both toy sets", state_color_dict_shades, fig_path)

fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_no_mom.png"
draw_state_distribution(flatten_pred_dict_cg['without_cg'], n_states, state_name_dict, "Without caregiver, both toy sets", state_color_dict_shades, fig_path)

fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_mobile_toy.png"
draw_state_distribution(flatten_pred_dict_toy_set['mobile'], n_states, state_name_dict, "Gross-motor toy sets", state_color_dict_shades, fig_path)

fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_stationary_toys.png"
draw_state_distribution(flatten_pred_dict_toy_set['fine'], n_states, state_name_dict, "Fine-motor toy sets", state_color_dict_shades, fig_path)
# %%
