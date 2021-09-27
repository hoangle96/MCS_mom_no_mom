#%%

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
from visualization import (
    draw_toy_state,
    draw_distribution,
    draw_timeline_with_merged_states,
    draw_state_distribution,
    draw_toy_state_with_std,
    draw_infant_each_min_matplotlib,
    draw_mean_state_locotion_across_conditions,
    draw_timeline_with_prob_to_check,
    draw_mean_state_locotion_across_conditions_separate_mean_std,
)
from hmm_model import (
    convert_to_int,
    convert_to_list_seqs,
    save_csv,
    init_hmm,
    rank_state,
)

import os
from pathlib import Path

# %%
feature_set = "n_new_toy_ratio"
no_ops_time = 10
n_states = 5
interval_length = 1.5
with open(
    "../data/interim/20210926"
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

# %%
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
    pred_dict_og = pickle.load(f)

#%%
pred_dict_og["MPS"][1]
# %%
model_file_name = "model_20210907_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)
print(state_name_dict)

#%%
for i in pred_dict_og["MPS"][1]:
    # print(i)
    print([state_name_dict[s] for s in pred_dict_og["MPS"][1][i]])
#%%
pred_dict["MPS"][1]
