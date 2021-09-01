# %%
import pickle
import numpy as np
import pomegranate as pom
import sys

from visualization import draw_distribution, draw_timeline_with_merged_states
from hmm_model import convert_to_int, convert_to_list_seqs, save_csv, init_hmm

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
import seaborn as sns

import itertools

from merge import merge_segment_with_state_calculation_all as merge_state


# %%
interval_length = 2
no_ops_time = 10
with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict_10s_2min = pickle.load(f)

with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
    time_arr_dict_10s_2min = pickle.load(f)

with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_label_"+str(interval_length)+"_min.pickle", 'rb') as f:
    labels_dict_10s_2min = pickle.load(f)
# %%
n_features = 4
shift_time_list = np.arange(0, interval_length, .25)

len_list_10s_2min = []

input_list_10s_2min = np.empty((0, n_features))
# input_list_7s_ = np.empty((0, n_features))

for task in tasks:
    for subj, shifted_df_dict in feature_dict_10s_2min[task].items():
        for shift_time, feature_vector in shifted_df_dict.items():
            # print(feature_vector)
            input_list_10s_2min = np.vstack((input_list_10s_2min, feature_vector))
            # input_list_7s_ = np.concatenate((input_list_7s_, feature_vector))

            len_list_10s_2min.append(len(feature_vector))

all_labels_10s_2min = []
for task in tasks:
    for subj, shifted_sequence in labels_dict_10s_2min[task].items():
        for shift_time, label in shifted_sequence.items(): 
            all_labels_10s_2min.append(label)
# %%
# cnt, val = np.histogram(input_list_10s_2min[:,0])
# cnt = cnt/cnt.sum()
# plt.style.use('seaborn')
# plt.bar(val[:-1], cnt, width = 3.2)
# plt.xlabel("# toys switches")
# plt.ylabel("%")
# plt.title("Distribution of toy switches after merging, small_no_ops threshold = 7s")
# plt.show()

toy_switch_bins = [0, 5, 10, 15]
n_bin_ep_rate = range(len(toy_switch_bins))
discretized_toy_switch_rate = np.digitize(input_list_10s_2min[:,0], toy_switch_bins, right = False)
_, counts = np.unique(discretized_toy_switch_rate, return_counts = True) 
heights = counts/counts.sum()
# fig = px.bar(x = n_bin_ep_rate, y = heights, text=counts)
# fig.update_layout(width=800, height=800, 
#                         title_text='Distribution of discretized rate of toy switch',
#                      xaxis = dict(
#                     tickmode = 'array',
#                     tickvals = toy_switch_bins,
#                 ))
# fig.show()

# %%
discretized_n_toys = np.where(input_list_10s_2min[:,1] > 4, 4, input_list_10s_2min[:,1])
n_toy_unique, counts = np.unique(discretized_n_toys, return_counts = True) 
heights = counts/counts.sum()
# fig = px.bar(x = n_toy_unique, y = heights, text=counts)

# fig.update_layout(width=800, height=800, 
#                         title_text='Distribution of discretized rate of number of toy',
#                      xaxis = dict(
#                     tickmode = 'array',
#                 ))
# fig.show()

# %%
discretized_n_new_toys = np.where(input_list_10s_2min[:,2] > 4, 4, input_list_10s_2min[:,2])
n_new_toys_unique, counts = np.unique(discretized_n_new_toys, return_counts = True) 
heights = counts/counts.sum()
# fig = px.bar(x = n_new_toys_unique, y = heights, text=counts)
# fig.update_layout(width=800, height=800, 
#                         title_text='Distribution of discretized rate of new toys',
#                      xaxis = dict(
#                     tickmode = 'array',
#                     # tickvals = ep_rate_dict[window_size],
#                 ))
# fig.show()

# %%
fav_toy_bin = [0, .2, .4, .6, .8]
n_bins_fav_toy = len(fav_toy_bin)

fav_toy_rate_discretized = np.digitize(input_list_10s_2min[:,3].copy(), fav_toy_bin, right = False)
_, counts = np.unique(fav_toy_rate_discretized, return_counts = True) 
heights = counts/counts.sum()
# fig = px.bar(x = fav_toy_bin, y = heights, text=counts)

# fig.update_layout(width=800, height=800, 
#                         title_text='Distribution of discretized rate of fav toy ratio',
#                      xaxis = dict(
#                     tickmode = 'array',
#                     tickvals =  fav_toy_bin,
#                 ))
# fig.show()

#%%
discretized_input_list_10s_2min = np.hstack((discretized_toy_switch_rate.reshape((-1,1)),\
                                    discretized_n_toys.reshape((-1,1)),\
                                    discretized_n_new_toys.reshape((-1,1)),\
                                    fav_toy_rate_discretized.reshape((-1,1))))
list_seq_10s_2min = convert_to_list_seqs(discretized_input_list_10s_2min, len_list_10s_2min)
list_seq_10s_2min = convert_to_int(list_seq_10s_2min)

# %%
SAVE = False
n_states = 5
seed = 1

model_10s_2min = init_hmm(n_states, discretized_input_list_10s_2min.T, seed)
model_10s_2min.bake()

# freeze the no_toys distribution so that its parameters are not updated. 
# "no_toys" state params are set so that all of the lowest bins = 0
for s in model_10s_2min.states:
    if s.name == "no_toys":
        for p in s.distribution.parameters[0]:
            p.frozen = True
model_10s_2min.fit(list_seq_10s_2min, labels = all_labels_10s_2min)

if SAVE:
    model_file_name = "model_20210726_"+str(interval_length)+"_interval_length_"+str(7)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
    model_file_path = Path('../models/20210726/')/model_file_name
    with open(model_file_path, 'wb+') as f:
        pickle.dump(model_10s_2min, f)

# %%
i = 0
input_dict_10s_2m = {}
for task in tasks:
    if task not in input_dict_10s_2m.keys():
        input_dict_10s_2m[task] = {}

    for subj, shifted_df_dict in feature_dict_10s_2min[task].items():
        if subj not in input_dict_10s_2m[task].keys():
            input_dict_10s_2m[task][subj] = {}


        for shift_time, feature_vector in shifted_df_dict.items():
            input_dict_10s_2m[task][subj][shift_time] = list_seq_10s_2min[i]
            i += 1

total_log_prob = 0
log_prob_list = []
pred_dict_10s_2min = {}
proba_dict_10s_2min = {}
all_proba_dict_10s_2min = {}

pred_by_task_10s_2min = {}
input_by_task_10s_2min = {}

for task in tasks:
    if task not in pred_dict_10s_2min.keys():
        pred_dict_10s_2min[task] = {}
        proba_dict_10s_2min[task] = {}
        all_proba_dict_10s_2min[task] = {}
        pred_by_task_10s_2min[task] = []
        input_by_task_10s_2min[task] = []

    for subj, shifted_dict in input_dict_10s_2m[task].items():
        if subj not in pred_dict_10s_2min[task].keys():
            pred_dict_10s_2min[task][subj] = {}
            proba_dict_10s_2min[task][subj] = {}
            all_proba_dict_10s_2min[task][subj] = {}

        for shift_time, feature_vector in shifted_dict.items():
            label= model_10s_2min.predict(feature_vector)
            pred_dict_10s_2min[task][subj][shift_time] = label
            pred_by_task_10s_2min[task].extend(label)
            input_by_task_10s_2min[task].extend(feature_vector)

            # if 4 in label:
                # print(feature_vector, label)
            proba_dict_10s_2min[task][subj][shift_time] = np.amax(model_10s_2min.predict_proba(feature_vector), axis = 1)
            log_prob = model_10s_2min.log_probability(feature_vector)
            all_proba_dict_10s_2min[task][subj][shift_time] = model_10s_2min.predict_proba(feature_vector)
            
            log_prob_list.append(log_prob)
print(np.mean(log_prob_list))
# %%

flatten_pred_dict_10s_2min = {}
flatten_pred_10s_2min = []
flatten_proba_dict_10s_2min = {}
for task in tasks:
    flatten_pred_dict_10s_2min[task] = []
    flatten_proba_dict_10s_2min[task] = []
    task_specific_pred_dict = pred_dict_10s_2min[task]
    for subj, subj_dict in task_specific_pred_dict.items():
        for shift_time, pred in subj_dict.items():
            flatten_pred_dict_10s_2min[task].extend(pred)
            flatten_proba_dict_10s_2min[task].extend(all_proba_dict_10s_2min[task][subj][shift_time])
            flatten_pred_10s_2min.extend(pred)

# %%
task_len = [len(flatten_pred_dict_10s_2min[task]) for task in tasks]
plt.style.use('seaborn')
x_ticks_dict = {0: ["[0, 5)", '[5, 10)', '[10, 15)', '[15+'], 1: ['0', '1', '2', '3', '4+'], 2: ['0', '1', '2', '3', '4+'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
feature_values = {0: range(1,5), 1: range(5), 2: range(5), 3: range(1,6)}

task_specific_input = convert_to_list_seqs(discretized_input_list_10s_2min, task_len)
task_specific_input_dict = {}
for i, task in enumerate(tasks):
    task_specific_input_dict[task] = task_specific_input[i]

# %%
state_name_dict = OrderedDict({4: "No_toys", 0: "F+", 3: "F", 1: "E", 2:"E+"})
feature_names = ["# toys switches", "# toys", "# new toys", 'fav toy ratio']


# %%
import importlib
# import inputs #import the module here, so that it can be reloaded.
# importlib.reload(visualization)
importlib.reload(sys.modules['visualization'])
importlib.reload(sys.modules['merge'])

# from modulename import func
from visualization import draw_distribution, draw_timeline_with_merged_states
from merge import merge_segment_with_state_calculation_all as merge_state


# %%
plt.style.use('seaborn')

draw_distribution(n_features, state_name_dict, discretized_input_list_10s_2min, np.array(flatten_pred_10s_2min), "all tasks",feature_names, x_ticks_dict, feature_values)


##############################################################
# 7 second lag #
#%% 
## test with different 

# %%
interval_length = 1.5
no_ops_time = 7
with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict_7s_1_5min = pickle.load(f)

with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
    time_arr_dict_7s_1_5min = pickle.load(f)

with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_label_"+str(interval_length)+"_min.pickle", 'rb') as f:
    labels_dict_7s_1_5min = pickle.load(f)
# %%
n_features = 4
shift_time_list = np.arange(0, interval_length, .25)

len_list_7s_1_5min = []

input_list_7s_1_5min = np.empty((0, n_features))
# input_list_7s_ = np.empty((0, n_features))

for task in tasks:
    for subj, shifted_df_dict in feature_dict_7s_1_5min[task].items():
        for shift_time, feature_vector in shifted_df_dict.items():
            # print(feature_vector)
            input_list_7s_1_5min = np.vstack((input_list_7s_1_5min, feature_vector))
            # input_list_7s_ = np.concatenate((input_list_7s_, feature_vector))

            len_list_7s_1_5min.append(len(feature_vector))

all_labels_7s_1_5min = []
for task in tasks:
    for subj, shifted_sequence in labels_dict_7s_1_5min[task].items():
        for shift_time, label in shifted_sequence.items(): 
            all_labels_7s_1_5min.append(label)
# %%

toy_switch_bins = [0, 5, 10, 15]
n_bin_ep_rate = range(len(toy_switch_bins))
discretized_toy_switch_rate = np.digitize(input_list_7s_1_5min[:,0], toy_switch_bins, right = False)
discretized_n_toys = np.where(input_list_7s_1_5min[:,1] > 4, 4, input_list_7s_1_5min[:,1])
discretized_n_new_toys = np.where(input_list_7s_1_5min[:,2] > 4, 4, input_list_7s_1_5min[:,2])
fav_toy_bin = [0, .2, .4, .6, .8]
n_bins_fav_toy = len(fav_toy_bin)

fav_toy_rate_discretized = np.digitize(input_list_7s_1_5min[:,3].copy(), fav_toy_bin, right = False)
discretized_input_list_7s_1_5min = np.hstack((discretized_toy_switch_rate.reshape((-1,1)),\
                                    discretized_n_toys.reshape((-1,1)),\
                                    discretized_n_new_toys.reshape((-1,1)),\
                                    fav_toy_rate_discretized.reshape((-1,1))))
list_seq_7s_1_5min = convert_to_list_seqs(discretized_input_list_7s_1_5min, len_list_7s_1_5min)
list_seq_7s_1_5min = convert_to_int(list_seq_7s_1_5min)
# %%
SAVE = False
n_states = 5
seed = 1

model_7s_1_5min = init_hmm(n_states, discretized_input_list_7s_1_5min.T, seed)
model_7s_1_5min.bake()

# freeze the no_toys distribution so that its parameters are not updated. 
# "no_toys" state params are set so that all of the lowest bins = 0
for s in model_7s_1_5min.states:
    if s.name == "no_toys":
        for p in s.distribution.parameters[0]:
            p.frozen = True
model_7s_1_5min.fit(list_seq_7s_1_5min, labels = all_labels_7s_1_5min)

if SAVE:
    model_file_name = "model_20210726_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
    model_file_path = Path('../models/20210726/')/model_file_name
    with open(model_file_path, 'wb+') as f:
        pickle.dump(model_7s_1_5min, f)

# %%
data = []

index_list = [[],[]]


features_obs_dict = {0: len(toy_switch_bins) , 1: 5, 2: 5, 3: 5}

for i in range(n_features):
    single_list = np.empty((features_obs_dict[i], n_states))
    for state_idx, state_i in enumerate(range(n_states)):
        observation_dict = model_7s_1_5min.states[state_i].distribution.parameters[0][i].parameters[0]
        for idx,k in enumerate(observation_dict.keys()):
            single_list[idx, state_idx] = np.round(observation_dict[k], 2)
    index_list[0].extend([i]*len(observation_dict.keys()))
    index_list[1].extend([i for i in observation_dict.keys()])

    data.extend(single_list)

tuples = list(zip(*index_list))
index = pd.MultiIndex.from_tuples(tuples, names=['feature', 'observation'])
df = pd.DataFrame(data, index = index, columns = ['state '+str(i) for i in range(n_states)])
file_path = Path('/scratch/mom_no_mom/reports/20210727/test/')
file_name = 'mean_'+str(n_states)+"_states_seed_"+str(seed)+"_1.5_min.csv"
save_csv(df, file_path, file_name)

# save the transition matrix for all

trans_matrix = pd.DataFrame(np.round(model_7s_1_5min.dense_transition_matrix()[:n_states+1,:n_states],2))
file_name = 'trans_matrix_'+str(n_states)+"_states_seed_"+str(seed)+'_1.5_min.csv'

save_csv(trans_matrix, file_path, file_name)


# %%

i = 0
input_dict_7s_1_5min = {}
for task in tasks:
    if task not in input_dict_7s_1_5min.keys():
        input_dict_7s_1_5min[task] = {}

    for subj, shifted_df_dict in feature_dict_7s_1_5min[task].items():
        if subj not in input_dict_7s_1_5min[task].keys():
            input_dict_7s_1_5min[task][subj] = {}


        for shift_time, feature_vector in shifted_df_dict.items():
            input_dict_7s_1_5min[task][subj][shift_time] = list_seq_7s_1_5min[i]
            i += 1

total_log_prob = 0
log_prob_list = []
pred_dict_7s_1_5min = {}
proba_dict_7s_1_5min = {}
all_proba_dict_7s_1_5min = {}

pred_by_task_7s_1_5min = {}
input_by_task_7s_1_5min = {}

for task in tasks:
    if task not in pred_dict_7s_1_5min.keys():
        pred_dict_7s_1_5min[task] = {}
        proba_dict_7s_1_5min[task] = {}
        all_proba_dict_7s_1_5min[task] = {}
        pred_by_task_7s_1_5min[task] = []
        input_by_task_7s_1_5min[task] = []

    for subj, shifted_dict in input_dict_7s_1_5min[task].items():
        if subj not in pred_dict_7s_1_5min[task].keys():
            pred_dict_7s_1_5min[task][subj] = {}
            proba_dict_7s_1_5min[task][subj] = {}
            all_proba_dict_7s_1_5min[task][subj] = {}

        for shift_time, feature_vector in shifted_dict.items():
            label= model_7s_1_5min.predict(feature_vector)
            pred_dict_7s_1_5min[task][subj][shift_time] = label
            pred_by_task_7s_1_5min[task].extend(label)
            input_by_task_7s_1_5min[task].extend(feature_vector)

            # if 4 in label:
                # print(feature_vector, label)
            proba_dict_7s_1_5min[task][subj][shift_time] = np.amax(model_7s_1_5min.predict_proba(feature_vector), axis = 1)
            log_prob = model_7s_1_5min.log_probability(feature_vector)
            all_proba_dict_7s_1_5min[task][subj][shift_time] = model_7s_1_5min.predict_proba(feature_vector)
            
            log_prob_list.append(log_prob)
print(np.mean(log_prob_list))


# %%

with open('../data/interim/20210726_'+str(7)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    pickle.dump(all_proba_dict_7s_1_5min, f)

with open('../data/interim/20210726_'+str(7)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    pickle.dump(pred_dict_7s_1_5min, f)


# %%

flatten_pred_dict_7s_1_5min = {}
flatten_pred_7s_1_5min = []
flatten_proba_dict_7s_1_5min = {}
for task in tasks:
    flatten_pred_dict_7s_1_5min[task] = []
    flatten_proba_dict_7s_1_5min[task] = []
    task_specific_pred_dict = pred_dict_7s_1_5min[task]
    for subj, subj_dict in task_specific_pred_dict.items():
        for shift_time, pred in subj_dict.items():
            flatten_pred_dict_7s_1_5min[task].extend(pred)
            flatten_proba_dict_7s_1_5min[task].extend(all_proba_dict_7s_1_5min[task][subj][shift_time])
            flatten_pred_7s_1_5min.extend(pred)

# %%
task_len = [len(flatten_pred_dict_7s_1_5min[task]) for task in tasks]
plt.style.use('seaborn')
x_ticks_dict = {0: ["[0, 5)", '[5, 10)', '[10, 15)', '[15+'], 1: ['0', '1', '2', '3', '4+'], 2: ['0', '1', '2', '3', '4+'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
feature_values = {0: range(1,5), 1: range(5), 2: range(5), 3: range(1,6)}

task_specific_input = convert_to_list_seqs(discretized_input_list_7s_1_5min, task_len)
task_specific_input_dict = {}
for i, task in enumerate(tasks):
    task_specific_input_dict[task] = task_specific_input[i]

# %%
state_name_dict = OrderedDict({4: "No_toys", 2: "F+", 0: "F", 3: "E", 1:"E+"})
feature_names = ["# toys switches", "# toys", "# new toys", 'fav toy ratio']

plt.style.use('seaborn')

draw_distribution(n_features, state_name_dict, discretized_input_list_7s_1_5min, np.array(flatten_pred_7s_1_5min), "all tasks",feature_names, x_ticks_dict, feature_values)


# %%
task = "MPS"
val, cnt = np.unique(np.array(flatten_pred_dict_7s_1_5min[task]).astype(int), return_counts = True)
pct = cnt/cnt.sum()
print(val, cnt, pct)

task_state_pct = {v: pct[idx] for idx, v in enumerate(val)}
for i in range(n_states):
    if i not in task_state_pct.keys():
        task_state_pct[i] = 0
for idx, state in enumerate(list(state_name_dict.keys())):
    plt.bar(idx, task_state_pct[state])
# plt.hist(np.array(flatten_pred_dict[task]).astype(int))
plt.ylim(top = .6)
plt.grid(b=None)
plt.xlabel('States', fontsize = 16)
plt.ylabel('% of all time in this condition', fontsize = 16)
plt.yticks(fontsize = 16)


plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"], fontsize = 16)
plt.title("With caregiver, Fine motor toys", fontsize = 16)
plt.show()

# %%
task = "MPM"
val, cnt = np.unique(np.array(flatten_pred_dict_7s_1_5min[task]).astype(int), return_counts = True)
pct = cnt/cnt.sum()
print(val, cnt)
task_state_pct = {v: pct[idx] for idx, v in enumerate(val)}
for i in range(n_states):
    if i not in task_state_pct.keys():
        task_state_pct[i] = 0
print(task_state_pct)
# task_state_pct[4] = 0
for idx, state in enumerate(list(state_name_dict.keys())):
    plt.bar(idx, task_state_pct[state])
# plt.hist(np.array(flatten_pred_dict[task]).astype(int))
plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"])
plt.ylim(top = .6)
plt.grid(b=None)
plt.xlabel('States', fontsize = 16)
plt.ylabel('% of all time in this condition', fontsize = 16)
plt.yticks(fontsize = 16)


plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"], fontsize = 16)
plt.title("With caregiver, Gross motor toys", fontsize = 16)
plt.show()

# %%
task = "NMS"
val, cnt = np.unique(np.array(flatten_pred_dict_7s_1_5min[task]).astype(int), return_counts = True)
pct = cnt/cnt.sum()
print(val, cnt)
task_state_pct = {v: pct[idx] for idx, v in enumerate(val)}
for i in range(n_states):
    if i not in task_state_pct.keys():
        task_state_pct[i] = 0
print(task_state_pct)
for idx, state in enumerate(list(state_name_dict.keys())):
    plt.bar(idx, task_state_pct[state])
# plt.hist(np.array(flatten_pred_dict[task]).astype(int))
plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"])
plt.ylim(top = .6)
plt.grid(b=None)
plt.xlabel('States', fontsize = 16)
plt.ylabel('% of all time in this condition', fontsize = 16)
plt.yticks(fontsize = 16)


plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"], fontsize = 16)
plt.title("Without caregiver, Fine motor toys", fontsize = 16)
plt.show()

# %%
task = "NMM"

val, cnt = np.unique(np.array(flatten_pred_dict_7s_1_5min[task]).astype(int), return_counts = True)
pct = cnt/cnt.sum()

print(val, cnt)
task_state_pct = {v: pct[idx] for idx, v in enumerate(val)}
for i in range(n_states):
    if i not in task_state_pct.keys():
        task_state_pct[i] = 0
for idx, state in enumerate(list(state_name_dict.keys())):
    plt.bar(idx, task_state_pct[state])
# plt.hist(np.array(flatten_pred_dict[task]).astype(int))
plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"])
plt.ylim(top = .6)
plt.grid(b=None)
plt.xlabel('States', fontsize = 16)
plt.ylabel('% of all time in this condition', fontsize = 16)
plt.yticks(fontsize = 16)


plt.xticks([0,1,2,3,4], ["No_ops", "F+", "F", "E", "E+"], fontsize = 16)
plt.title("Without caregiver, Gross motor toys", fontsize = 16)
plt.show()

# %%
with open('../data/interim/20210726_7_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    task_to_storing_dict = pickle.load(f)

with open('../data/interim/20210726_'+str(no_ops_time)+'_no_ops_threshold_floor_time.pickle', 'rb') as f:
    floor_time = pickle.load(f)

with open("../data/interim/20210726_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
    time_arr_dict = pickle.load(f)

subj_list = list(task_to_storing_dict['MPS'].keys())
shift_time_list = np.arange(0,interval_length,0.25)

merged_pred_dict_all = {}
merged_proba_dict_all = {}
time_subj_dict_all = {}
for task in tasks:
    print(task)
    merged_df_dict = task_to_storing_dict[task]
    time_arr_shift_dict = time_arr_dict[task]
    pred_subj_dict = pred_dict_7s_1_5min[task]
    prob_subj_dict = all_proba_dict_7s_1_5min[task]

    merged_pred_dict_all_task_specific, merged_proba_dict_all_task_specific, time_subj_dict_all_task_specific = merge_state(subj_list, shift_time_list, merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size = interval_length, n_states = 5, shift_interval = 60000*.25)

    merged_pred_dict_all[task] = merged_pred_dict_all_task_specific
    merged_proba_dict_all[task] = merged_proba_dict_all_task_specific
    time_subj_dict_all[task] = time_subj_dict_all_task_specific

with open('../data/interim/20210726_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    pickle.dump(merged_pred_dict_all, f)

with open('../data/interim/20210726_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    pickle.dump(merged_proba_dict_all, f)

with open('../data/interim/20210726_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    pickle.dump(time_subj_dict_all, f)


# %%
for subj in subj_list:
    for task in tasks:
        df = pd.DataFrame()
        for df_ in task_to_storing_dict[task][subj]:
            df = pd.concat([df, df_])
        fig_name = '../figures/hmm/20210721/window_size_1.5/'+str(n_states)+'_states/'+task+'/'+str(subj)+".png"
        draw_timeline_with_merged_states(subj, df, merged_pred_dict_all[task][subj], time_subj_dict_all[task][subj], state_name_dict, fig_name= fig_name, gap_size = 60000*.25, show=False)


#%%
# Combine with toys
def merge_toy_pred(pred_df, subj_df):
    toys = []

    all_onsets = list(set(pred_df['onset'].unique().tolist() + subj_df['onset'].unique().tolist()))
    # print(all_onsets)
    all_offsets = list(set(pred_df['offset'].unique().tolist() + subj_df['offset'].unique().tolist()))
    # print(len(all_onsets) - len(all_offsets))
    time = list(set(all_onsets + all_offsets))
    time.sort()
    toys_list = []
    pred_list = []
    onset_list = []
    offset_list = []
    for idx, onset in enumerate(time):
        if idx != len(time) - 1:
            offset = time[idx+1] 
            onset_list.append(onset)
            offset_list.append(offset)

            pred = pred_df.loc[(pred_df.loc[:,'onset'] <= onset) & (pred_df.loc[:,'offset'] >= offset), 'pred'].tolist()
            pred_list.append(pred[0])
            toys = subj_df.loc[(subj_df.loc[:,'onset'] <= onset) & (subj_df.loc[:,'offset'] >= offset), 'toy'].tolist()
            if 'no_ops' in toys:
                toys = [t for t in toys if 'no_ops' not in t]
            toys_list.append(list(set(itertools.chain.from_iterable(toys))))
    return pd.DataFrame({'onset': onset_list, 'offset': offset_list, 'pred': pred_list, 'toys':toys_list })


# %%
subj_list = list(task_to_storing_dict['MPS'].keys())

toy_pred_list = {}
    
for task in tasks:
    toy_pred_list[task] = {}
    for subj in subj_list:
        subj_df = pd.DataFrame()
        pred = []
        onset = []
        offset = []

        onset.append(time_subj_dict_all[task][subj][0] - shift_time_list[1]*60000)
        onset.extend(time_subj_dict_all[task][subj][:-1]) 
        offset.extend(time_subj_dict_all[task][subj])
        pred.extend(merged_pred_dict_all[task][subj])

        for df_ in task_to_storing_dict[task][subj]:
            subj_df = pd.concat([subj_df, df_])
        if len(onset) == len(pred):
            pred_df = pd.DataFrame(data = {'onset': onset, 'offset': offset, 'pred': pred})

            pred_df = merge_toy_pred(pred_df, subj_df)
            toy_pred_list[task][subj] = pred_df

# %%
toy_colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'violet', 'bucket': 'navy'}

stationary_df = pd.DataFrame()
for task in ['MPS', "NMS"]:
    for subj in subj_list:
        if subj in toy_pred_list[task].keys():
            stationary_df = pd.concat([stationary_df,  toy_pred_list[task][subj]])

mobile_df = pd.DataFrame()
for task in ['MPM', "NMM"]:
    for subj in subj_list:
        if subj in toy_pred_list[task].keys():
            mobile_df = pd.concat([mobile_df,  toy_pred_list[task][subj]])

nmm_df = pd.DataFrame()
for subj in subj_list:
    if subj in toy_pred_list["NMM"].keys():
        nmm_df = pd.concat([nmm_df, toy_pred_list["NMM"][subj]])

nms_df = pd.DataFrame()
for subj in subj_list:
    if subj in toy_pred_list["NMS"].keys():
        nms_df = pd.concat([nms_df, toy_pred_list["NMS"][subj]])

mps_df = pd.DataFrame()
for subj in subj_list:
    if subj in toy_pred_list["MPS"].keys():
        mps_df = pd.concat([mps_df, toy_pred_list["MPS"][subj]])

mpm_df = pd.DataFrame()
for subj in subj_list:
    if subj in toy_pred_list["MPM"].keys():
        mpm_df = pd.concat([mpm_df, toy_pred_list["MPM"][subj]])
# %%
stationary_df = stationary_df.explode('toys') 
stationary_df['duration'] = stationary_df['offset'] - stationary_df['onset'] 
stationary_df['pred'] = stationary_df['pred'].replace(state_name_dict)

stationary_toy_to_pred_dict = (stationary_df.groupby(['pred', 'toys'])['duration'].sum()/stationary_df.groupby(['pred'])['duration'].sum()).to_dict()
stationary_toy_list = stationary_df['toys'].dropna().unique()
# stationary_toy_list = np.unique(stationary_toy_list[~np.isnan(stationary_toy_list)])
plt.style.use("seaborn")
fig = plt.figure(figsize= (15,8))
for x_loc, state in enumerate(state_name_dict.values()):
    if state != "No_toys":
        current_state_dict_stationary_toy = {k: stationary_toy_to_pred_dict[k] for k in stationary_toy_to_pred_dict.keys() if state in k}
        for idx, toy in enumerate(stationary_toy_list):
            key = (state, toy)
            if x_loc == 1:
                plt.bar(x_loc*8 + idx, current_state_dict_stationary_toy[key], label = toy, color = toy_colors_dict[toy])
            else:
                plt.bar(x_loc*8 + idx, current_state_dict_stationary_toy[key], color = toy_colors_dict[toy])
plt.xticks([10.5, 18.5, 26.5, 34.5], ['F+', 'F', 'E', 'E+'], fontsize = 16)
plt.ylabel("% time in each state playing with toys", fontsize = 16)

plt.xlabel('States', fontsize = 16)
plt.ylim(top = 0.5)
plt.yticks(fontsize = 16)
plt.title("Both conditions, fine motor toys", fontsize = 16)

plt.legend(fontsize = 16)
plt.show()

# %%
mobile_df = mobile_df.explode('toys') 
mobile_df['duration'] = mobile_df['offset'] - mobile_df['onset'] 
mobile_df['pred'] = mobile_df['pred'].replace(state_name_dict)

mobile_toy_to_pred_dict = (mobile_df.groupby(['pred', 'toys'])['duration'].sum()/mobile_df.groupby(['pred'])['duration'].sum()).to_dict()

mobile_toy_list = mobile_df['toys'].dropna().unique()
# stationary_toy_list = np.unique(stationary_toy_list[~np.isnan(stationary_toy_list)])
plt.style.use("seaborn")
fig = plt.figure(figsize= (15,8))
for x_loc, state in enumerate(state_name_dict.values()):
    if state != "No_toys":
        current_state_dict_mobile_toy = {k: mobile_toy_to_pred_dict[k] for k in mobile_toy_to_pred_dict.keys() if state in k}
        for idx, toy in enumerate(mobile_toy_list):
            key = (state, toy)
            if key not in mobile_toy_to_pred_dict.keys():
                val = 0
            else:
                val = current_state_dict_mobile_toy[key]
            if x_loc == 1:
                plt.bar(x_loc*8 + idx, val, label = toy, color = toy_colors_dict[toy])
            else:
                plt.bar(x_loc*8 + idx, val, color = toy_colors_dict[toy])
plt.xticks([10.5, 18.5, 26.5, 34.5], ['F+', 'F', 'E', 'E+'], fontsize = 16)
plt.ylabel("% time in each state playing with toys", fontsize = 16)
plt.ylim(top = 0.5)

plt.xlabel('States', fontsize = 16)
plt.yticks(fontsize = 16)
plt.title("Both conditions, gross motor toys", fontsize = 16)

plt.legend(fontsize = 16)
plt.show()

# %%
mpm_df = mpm_df.explode('toys') 
mpm_df['duration'] = mpm_df['offset'] - mpm_df['onset'] 
mpm_df['pred'] = mpm_df['pred'].replace(state_name_dict)

mpm_toy_to_pred_dict = (mpm_df.groupby(['pred', 'toys'])['duration'].sum()/mpm_df.groupby(['pred'])['duration'].sum()).to_dict()

mpm_toy_list = mpm_df['toys'].dropna().unique()
# stationary_toy_list = np.unique(stationary_toy_list[~np.isnan(stationary_toy_list)])
plt.style.use("seaborn")
fig = plt.figure(figsize= (12,15))
for x_loc, state in enumerate(state_name_dict.values()):
    if state != "No_toys":
        current_state_dict_mpm_toy = {k: mpm_toy_to_pred_dict[k] for k in mpm_toy_to_pred_dict.keys() if state in k}
        for idx, toy in enumerate(mpm_toy_list):
            key = (state, toy)
            if key not in mpm_toy_to_pred_dict.keys():
                val = 0
            else:
                val = current_state_dict_mpm_toy[key]
            if x_loc == 1:
                plt.bar(x_loc*8 + idx, val, label = toy, color = toy_colors_dict[toy])
            else:
                plt.bar(x_loc*8 + idx, val, color = toy_colors_dict[toy])
plt.xticks([10.5, 18.5, 26.5, 34.5], ['F+', 'F', 'E', 'E+'], fontsize = 16)
plt.ylabel("% time in each state playing with toys", fontsize = 16)
plt.ylim(top = 0.5)

plt.xlabel('States', fontsize = 16)
plt.yticks(fontsize = 16)
plt.title("With caregiver, gross motor toys", fontsize = 16)

plt.legend(fontsize = 16)
plt.show()

# %%
nmm_df = nmm_df.explode('toys') 
nmm_df['duration'] = nmm_df['offset'] - nmm_df['onset'] 
nmm_df['pred'] = nmm_df['pred'].replace(state_name_dict)

nmm_toy_to_pred_dict = (nmm_df.groupby(['pred', 'toys'])['duration'].sum()/nmm_df.groupby(['pred'])['duration'].sum()).to_dict()

nmm_toy_list = nmm_df['toys'].dropna().unique()
# stationary_toy_list = np.unique(stationary_toy_list[~np.isnan(stationary_toy_list)])
plt.style.use("seaborn")
fig = plt.figure(figsize= (12,15))
for x_loc, state in enumerate(state_name_dict.values()):
    if state != "No_toys":
        current_state_dict_nmm_toy = {k: nmm_toy_to_pred_dict[k] for k in nmm_toy_to_pred_dict.keys() if state in k}
        for idx, toy in enumerate(nmm_toy_list):
            key = (state, toy)
            if key not in nmm_toy_to_pred_dict.keys():
                val = 0
            else:
                val = current_state_dict_nmm_toy[key]
            if x_loc == 1:
                plt.bar(x_loc*8 + idx, val, label = toy, color = toy_colors_dict[toy])
            else:
                plt.bar(x_loc*8 + idx, val, color = toy_colors_dict[toy])
plt.xticks([10.5, 18.5, 26.5, 34.5], ['F+', 'F', 'E', 'E+'], fontsize = 16)
plt.ylabel("% time in each state playing with toys", fontsize = 16)

plt.xlabel('States', fontsize = 16)
plt.yticks(fontsize = 16)
plt.title("Without caregiver, gross motor toys", fontsize = 16)
plt.ylim(top = 0.5)

plt.legend(fontsize = 16)
plt.show()

# %%
mps_df = mps_df.explode('toys') 
mps_df['duration'] = mps_df['offset'] - mps_df['onset'] 
mps_df['pred'] = mps_df['pred'].replace(state_name_dict)

mps_toy_to_pred_dict = (mps_df.groupby(['pred', 'toys'])['duration'].sum()/mps_df.groupby(['pred'])['duration'].sum()).to_dict()

mps_toy_list = mps_df['toys'].dropna().unique()
# stationary_toy_list = np.unique(stationary_toy_list[~np.isnan(stationary_toy_list)])
plt.style.use("seaborn")
fig = plt.figure(figsize= (12,15))
for x_loc, state in enumerate(state_name_dict.values()):
    if state != "No_toys":
        current_state_dict_mps_toy = {k: mps_toy_to_pred_dict[k] for k in mps_toy_to_pred_dict.keys() if state in k}
        for idx, toy in enumerate(mps_toy_list):
            key = (state, toy)
            if key not in mps_toy_to_pred_dict.keys():
                val = 0
            else:
                val = current_state_dict_mps_toy[key]
            if x_loc == 1:
                plt.bar(x_loc*8 + idx, val, label = toy, color = toy_colors_dict[toy])
            else:
                plt.bar(x_loc*8 + idx, val, color = toy_colors_dict[toy])
plt.xticks([10.5, 18.5, 26.5, 34.5], ['F+', 'F', 'E', 'E+'], fontsize = 16)
plt.ylabel("% time in each state playing with toys", fontsize = 16)

plt.xlabel('States', fontsize = 16)
plt.yticks(fontsize = 16)
plt.title("With caregiver, fine motor toys", fontsize = 16)
plt.ylim(top = 0.5)

plt.legend(fontsize = 16)
plt.show()

# %%
nms_df = nms_df.explode('toys') 
nms_df['duration'] = nms_df['offset'] - nms_df['onset'] 
nms_df['pred'] = nms_df['pred'].replace(state_name_dict)

nms_toy_to_pred_dict = (nms_df.groupby(['pred', 'toys'])['duration'].sum()/nms_df.groupby(['pred'])['duration'].sum()).to_dict()

nms_toy_list = nms_df['toys'].dropna().unique()
# stationary_toy_list = np.unique(stationary_toy_list[~np.isnan(stationary_toy_list)])
plt.style.use("seaborn")
fig = plt.figure(figsize= (12,15))
for x_loc, state in enumerate(state_name_dict.values()):
    if state != "No_toys":
        current_state_dict_nms_toy = {k: nms_toy_to_pred_dict[k] for k in nms_toy_to_pred_dict.keys() if state in k}
        for idx, toy in enumerate(nms_toy_list):
            key = (state, toy)
            if key not in nms_toy_to_pred_dict.keys():
                val = 0
            else:
                val = current_state_dict_nms_toy[key]
            if x_loc == 1:
                plt.bar(x_loc*8 + idx, val, label = toy, color = toy_colors_dict[toy])
            else:
                plt.bar(x_loc*8 + idx, val, color = toy_colors_dict[toy])
plt.xticks([10.5, 18.5, 26.5, 34.5], ['F+', 'F', 'E', 'E+'], fontsize = 16)
plt.ylabel("% time in each state playing with toys", fontsize = 16)

plt.xlabel('States', fontsize = 16)
plt.yticks(fontsize = 16)
plt.title("Without caregiver, fine motor toys", fontsize = 16)
plt.ylim(top = 0.5)

plt.legend(fontsize = 16)
plt.show()