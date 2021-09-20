# %%
from scipy import stats
import numpy as np
import pandas as pd
from variables import (
    tasks,
    condition_name,
    state_color_dict,
    stationary_toys_list,
    mobile_toys_list,
    state_color_dict_shades,
)
import pickle
from merge import merge_segment_with_state_calculation_all
import os
from pathlib import Path
from all_visualization_20210824 import rank_state
import matplotlib.pyplot as plt
from scipy.stats import linregress
# %%
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = "n_new_toy_ratio"

model_file_name = "model_20210907_" + feature_set + "_" + \
    str(interval_length) + "_interval_length_" + str(no_ops_time) + \
    "_no_ops_threshold_" + str(n_states) + "_states.pickle"
model_file_path = Path("../models/hmm/20210907/" +
                       feature_set) / model_file_name
with open(model_file_path, "rb") as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

with open('../data/interim/20210818_baby_info.pickle', 'rb') as f:
    infant_info = pickle.load(f)

with open("../data/interim/20210907" + feature_set + "_" + str(no_ops_time) + "_no_ops_threshold" + str(n_states) + "_states_all_pred_prob_" + str(interval_length) + "_min.pickle", "rb") as f:
    all_prob_dict_all = pickle.load(f)
with open("../data/interim/20210907" + feature_set + "_" + str(no_ops_time) + "_no_ops_threshold" + str(n_states) + "_states_merged_prediction_" + str(interval_length) + "_min.pickle", "rb") as f:
    merged_pred_dict_all = pickle.load(f)

with open('../data/interim/20210718_babymovement.pickle', 'rb') as f:
    baby_movement = pickle.load(f)

total_num_steps_dict = {}
rank_infant_by_step_take_each_task = {}
# all_step =
for task_idx, (task, movement_dict) in enumerate(baby_movement.items()):
    each_task_infant_order = []
    each_task_dict = {}
    for subj, movement_df in movement_dict.items():
        each_task_dict[subj] = movement_df['babymovementSteps'].sum()
    total_num_steps_dict[task] = each_task_dict
# %%
convert_state_dict = {}
time_in_explore_states = {}
time_in_focus_states = {}
for task in tasks:
    convert_state_dict_subj = {}
    time_in_explore_states_subj = {}
    time_in_focus_states_subj = {}
    for subj, seq in merged_pred_dict_all[task].items():
        named_seq = [state_name_dict[i] for i in seq]
        convert_state_dict_subj[subj] = named_seq

        state, cnt = np.unique(named_seq, return_counts=True)

        total_explore_time = (cnt[state == "3"] + cnt[state == "4"])/cnt.sum()
        if len(total_explore_time) == 0:
            total_explore_time = 0
        else:
            total_explore_time = total_explore_time[0]
        time_in_explore_states_subj[subj] = total_explore_time

        total_focus_time = (cnt[state == "1"] + cnt[state == "2"])/cnt.sum()
        if len(total_focus_time) == 0:
            total_focus_time = 0
        else:
            total_focus_time = total_focus_time[0]
        time_in_focus_states_subj[subj] = total_focus_time

    convert_state_dict[task] = convert_state_dict_subj
    time_in_focus_states[task] = time_in_focus_states_subj
    time_in_explore_states[task] = time_in_explore_states_subj

# %%
infant_info.keys()
# %%
plt.style.use("seaborn")
fig = plt.figure(figsize=(10, 10), facecolor='white')
facecolor_dict_by_task = {"MPS": "red",
                          "NMS": 'none', "MPM": "blue", "NMM": 'none'}
edgecolor_dict_by_task = {"MPS": "red",
                          "NMS": "red", "MPM": "blue", "NMM": "blue"}
plt.facecolor = 'white'
# ax.set_facecolor('white')
for task in tasks:
    if task == "NMM":
        x = np.array(list(infant_info['walking_exp'].values()))
        y = list(time_in_explore_states[task].values())
        plt.scatter(x, y,
                    facecolors=facecolor_dict_by_task[task], edgecolors=edgecolor_dict_by_task[task],
                    linewidths=2)

        # x_ = np.array(x)[x != 0]
        # y_ = np.array(y)[x != 0]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print(r_value**2)
    else:
        plt.scatter(np.array(list(infant_info['walking_exp'].values())), time_in_explore_states[task].values(),
                    facecolors=facecolor_dict_by_task[task], edgecolors=edgecolor_dict_by_task[task],
                    linewidths=2, alpha=0.1)

# %%
x = np.array(list(infant_info['walking_exp'].values()))
y = np.array(list(time_in_explore_states["MPM"].values()))
x_ = x[x >= 90]
y_ = y[x >= 90]

print(x)
print(y)
print(len(x_))
print(len(y_))
slope, intercept, r_value, p_value, std_err = linregress(x_, y_)
print(r_value**2)

# %%
novice_time_in_explore = {}
experienced_time_in_explore = {}
novice_time_in_focus = {}
experienced_time_in_focus = {}
for task in tasks:
    novice_time_in_explore[task] = []
    experienced_time_in_explore[task] = []
    experienced_time_in_focus[task] = []
    novice_time_in_focus[task] = []

    for infant_id, walking_exp in infant_info['walking_exp'].items():
        if walking_exp >= 90:
            experienced_time_in_explore[task].append(
                time_in_explore_states[task][infant_id])
            experienced_time_in_focus[task].append(
                time_in_focus_states[task][infant_id])
            # experienced_time_in_explore[task].append(time_in_focus_states[task][infant_id])
        else:
            novice_time_in_explore[task].append(
                time_in_explore_states[task][infant_id])
            novice_time_in_focus[task].append(
                time_in_focus_states[task][infant_id]
            )
            # novice_time_in_focus[task].append(time_in_focus_states[task][infant_id])
            
# %%
len(novice_time_in_explore['MPM'])

# %%
value, t_test = stats.ttest_ind(
    experienced_time_in_explore['MPM'], novice_time_in_explore['MPM'])
print(value, t_test/2)
value, t_test = stats.ttest_ind(
    experienced_time_in_explore['NMM'], novice_time_in_explore['NMM'])
print(value, t_test/2)

value, t_test = stats.ttest_ind(
    experienced_time_in_explore['NMS'], novice_time_in_explore['NMS'])
print(value, t_test/2)

value, t_test = stats.ttest_ind(
    experienced_time_in_explore['MPS'], novice_time_in_explore['MPS'])
print(value, t_test/2)