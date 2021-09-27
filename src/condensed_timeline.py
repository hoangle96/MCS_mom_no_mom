#%%
import numpy as np
from all_visualization_20210824 import rank_state
from merge import merge_segment_with_state_calculation_all
from merge import merge_segment_with_state_calculation_all, merge_toy_pred
import sys
import importlib
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from collections.abc import Hashable
import pickle
from variables import (
    toys_dict,
    tasks,
    toys_list,
    mobile_toys_list,
    toys_of_interest_dict,
    small_no_ops_threshold_dict,
    condition_name,
    state_color_dict,
)
from pathlib import Path
import pandas as pd
from visualization import draw_comparison, draw_timeline_with_prob_to_check

from pathlib import Path
import matplotlib.pyplot as plt
import pomegranate as pom
from matplotlib.patches import Rectangle, Patch
import matplotlib

#%%
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = "n_new_toy_ratio"
# model_file_name = "model_20210824_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_name = (
    "model_20210907_"
    + feature_set
    + "_"
    + str(interval_length)
    + "_interval_length_"
    + str(no_ops_time)
    + "_no_ops_threshold_"
    + str(n_states)
    + "_states.pickle"
)

model_file_path = Path("../models/hmm/20210907/" + feature_set) / model_file_name

with open(model_file_path, "rb") as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

with open(model_file_path, "rb") as f:
    model = pickle.load(f)
with open(
    "../data/interim/20210907"
    + feature_set
    + "_"
    + str(no_ops_time)
    + "_no_ops_threshold_"
    + str(n_states)
    + "_states_prediction_"
    + str(interval_length)
    + "_min.pickle",
    "rb",
) as f:
    pred_dict = pickle.load(f)

with open(
    "../data/interim/20210907"
    + feature_set
    + "_"
    + str(no_ops_time)
    + "_no_ops_threshold_"
    + str(n_states)
    + "_states_prediction_all_prob_"
    + str(interval_length)
    + "_min.pickle",
    "rb",
) as f:
    all_proba_dict = pickle.load(f)

with open(
    "../data/interim/20210907"
    + feature_set
    + "_"
    + str(no_ops_time)
    + "_no_ops_threshold"
    + str(n_states)
    + "_states_merged_prediction_"
    + str(interval_length)
    + "_min.pickle",
    "rb",
) as f:
    merged_pred_dict_all = pickle.load(f)

with open(
    "../data/interim/20210907"
    + feature_set
    + "_"
    + str(no_ops_time)
    + "_no_ops_threshold"
    + str(n_states)
    + "_states_merged_prediction_prob_"
    + str(interval_length)
    + "_min.pickle",
    "rb",
) as f:
    merged_proba_dict_all = pickle.load(f)

# with open('..data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
#     time_subj_dict_all = pickle.load(f)

with open(
    "../data/interim/20210824_"
    + str(no_ops_time)
    + "_no_ops_threshold_clean_data_for_feature_engineering.pickle",
    "rb",
) as f:
    task_to_storing_dict = pickle.load(f)

with open("../data/interim/20210818_baby_info.pickle", "rb") as f:
    infant_info = pickle.load(f)

with open("../data/interim/20210718_babymovement.pickle", "rb") as f:
    baby_movement = pickle.load(f)

#%%
# timeline with number of steps
total_num_steps_dict = {}
for task, movement_dict in baby_movement.items():
    each_task_dict = {}
    for subj, movement_df in movement_dict.items():
        each_task_dict[subj] = movement_df["babymovementSteps"].sum()
    ranked_step_dict = {
        k: v for k, v in sorted(each_task_dict.items(), key=lambda item: item[1])
    }
    total_num_steps_dict[task] = ranked_step_dict

all_n_steps = []
for task in tasks:
    all_n_steps.extend(list(total_num_steps_dict[task].values()))
max_ = np.amax(all_n_steps)
for task in tasks:
    # task = "MPS"
    fig_name = (
        "../figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states/"
        + "condensed_timelines_"
        + task
        + "_by_steps.png"
    )
    state_color_dict_shades = {
        "0": "gainsboro",
        "1": "maroon",
        "2": "salmon",
        "3": "royalblue",
        "4": "midnightblue",
        "5": "midnightblue",
        "6": "midnightblue",
        "7": "blue",
    }
    ranked_step_dict = total_num_steps_dict[task]
    fig, axs = plt.subplots(
        nrows=len(ranked_step_dict.keys()), ncols=2, figsize=(100, 20)
    )
    # print(axs.shape)
    # plt.suptitle("Condensed state sequences, "+condition_name[task]+",\ninfants ranked by increasing walking experience", fontsize = 52)
    for ax_id, infant_id in enumerate(ranked_step_dict.keys()):
        state_list = [state_name_dict[i] for i in merged_pred_dict_all[task][infant_id]]
        color_list = [
            state_color_dict_shades[state_list[i]] for i in range(len(state_list))
        ]
        session_len = len(state_list) if len(state_list) <= 16 else 16
        # print(color_list)
        for i in range(session_len):
            # print(state_color_dict_shades[state_list[i]])
            # print(i)
            axs[ax_id][0].add_patch(
                Rectangle(
                    (i, 0),
                    1,
                    5,
                    ec="black",
                    fc=state_color_dict_shades[state_list[i]],
                    fill=True,
                    alpha=0.7,
                )
            )
            axs[ax_id][0].set_xticks(np.arange(0, 18, 2))
            axs[ax_id][0].set_xticklabels("")

            # axs[ax_id][0].set_xticklabels([str(x) for x in np.arange(0, 9, 1)])
            # axs[ax_id].set_ylabel(str(ranked_infant_dict[infant_id]), color = infant_group_color[infant_group], fontsize = 'xx-large')
            axs[ax_id][0].set_yticklabels("")
        axs[ax_id][0].set_xlim(right=16)

        # axs[ax_id][0].yaxis.set_label_position("right")
        axs[ax_id][1].barh(0, ranked_step_dict[infant_id])
        axs[ax_id][1].set_xlim(right=max_)
        axs[ax_id][1].set_ylabel("")
        axs[ax_id][1].set_yticklabels("")

        if ax_id != len(ranked_step_dict.keys()) - 1:
            axs[ax_id][1].set_xticklabels("")

    # legend_elements = []
    # for state in range(n_states):
    #     legend_elements.append(Patch(facecolor=state_color_dict_shades[str(state)],\
    #                                  edgecolor=state_color_dict_shades[str(state)],alpha = 0.7,\
    #                                  label=str(state)))

    # axs[ax_id].legend(handles=legend_elements, loc = 1, bbox_to_anchor=(1.05, 1), fontsize = 36, ncol=n_states)
    axs[ax_id][0].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize=52)
    axs[ax_id][0].set_xlabel("Minutes", fontsize=52)
    axs[ax_id][1].set_xlabel("Total # of steps", fontsize=52)

    params = {"axes.labelsize": 24, "xtick.labelsize": 52}
    matplotlib.rcParams.update(params)
    # plt.tight_layout()
    plt.savefig(fig_name)

#%%
# rank infants by walking experience
condition_name = {"MPS": "With caregiver, fine motor toys",
                  "NMS": "Without caregiver, fine motor toys",
                  "NMM": "Without caregiver, gross motor toys",
                  "MPM": "With caregiver, gross motor toys"
                  }
ranked_infant_by_walk_exp = {}
for infant_id, walking_exp in infant_info['walking_exp'].items():
    ranked_infant_by_walk_exp[infant_id] = walking_exp//30

ranked_infant_dict = {k: v for k, v in sorted(
    ranked_infant_by_walk_exp.items(), key=lambda item: item[1])}
# %%
infant_group_color = {0: "darkblue", 1: 'darkgreen', 2: 'darkred'}
# print(ranked_infant_dict)\\
for task in tasks:
    fig_name = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_' + \
        str(no_ops_time)+'/window_size_'+str(interval_length)+'/' + \
        str(n_states)+'_states/'+"condensed_timelines_"+task+".png"
    state_color_dict_shades = {"0": 'gainsboro',  "1": 'maroon', "2": 'salmon', "3": 'royalblue',
                               "4": 'midnightblue',  "5": 'midnightblue', "6": 'midnightblue', "7": 'blue'}
    fig, axs = plt.subplots(
        nrows=len(ranked_infant_dict.keys()), ncols=1, figsize=(40, 15), sharex=True)
    # plt.suptitle("Condensed state sequences, "+condition_name[task]+",\ninfants ranked by increasing walking experience", fontsize = 52)
    for ax_id, infant_id in enumerate(ranked_infant_dict.keys()):
        if ranked_infant_dict[infant_id] < 3:
            infant_group = 0
        elif ranked_infant_dict[infant_id] < 6:
            infant_group = 1
        else:
            infant_group = 2

        state_list = [state_name_dict[i]
                      for i in merged_pred_dict_all[task][infant_id]]
        session_len = len(state_list) if len(state_list) <= 16 else 16
        for i in range(session_len):
            axs[ax_id].add_patch(Rectangle(
                (i, 0), 1, 5, ec='black', fc=state_color_dict_shades[state_list[i]], fill=True, alpha=0.7))
            axs[ax_id].set_xticks(np.arange(0, 18, 2))
            axs[ax_id].set_xticklabels([str(x) for x in np.arange(0, 9, 1)])
            axs[ax_id].set_ylabel(str(ranked_infant_dict[infant_id]),
                                  color=infant_group_color[infant_group], fontsize='xx-large')
            axs[ax_id].set_yticklabels('')
            axs[ax_id].yaxis.set_label_position("right")

    legend_elements = []
    for state in range(n_states):
        legend_elements.append(Patch(facecolor=state_color_dict_shades[str(state)],
                                     edgecolor=state_color_dict_shades[str(
                                         state)], alpha=0.7,
                                     label=str(state)))

    # axs[ax_id].legend(handles=legend_elements, loc = 1, bbox_to_anchor=(1.05, 1), fontsize = 36, ncol=n_states)
    axs[ax_id].set_xticklabels([str(x)
                                for x in np.arange(0, 9, 1)], fontsize=48)
    axs[ax_id].set_xlabel("Minutes", fontsize=48)
    params = {'axes.labelsize': 18}
    matplotlib.rcParams.update(params)
    # plt.tight_layout()
    plt.savefig(fig_name)