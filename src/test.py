#%%
import numpy as np 
import pandas as pd 
from collections.abc import Hashable
import pickle
from variables import toys_dict, tasks, toys_list, mobile_toys_list, toys_of_interest_dict, small_no_ops_threshold_dict, condition_name, state_color_dict
import cv2
from pathlib import Path 
import glob 
import pandas as pd 
from visualization import draw_comparison, draw_timeline_with_prob_to_check

from pathlib import Path
import matplotlib.pyplot as plt
import pomegranate as pom
from matplotlib.patches import Rectangle, Patch

# from all_visualization import rank_state

#%%
states = np.array([1,2,3])
cnt = np.array([0.3, 0.2, 0.5])

print(cnt[states == 1])

# %%
bins = [0, 30, 60, 90, 120, 150, 180, 240, 270, 300]
bins_3 = np.arange(0, 181, 90)
print(bins_3)
x = np.array([33, 30, 12, 121, 120, 181, 305])
print(np.digitize(x, bins_3, right = False))
#%%
# print(isinstance("no_ops", Hashable))
# n_states_6 = [-24.10542593139329,-23.42322274430872, -30.25212157223115,-23.07387727085122, -22.855473735190806] 
# #[-24.10542593139329, -23.42322274430872]
# # [-23.432169840574453, -24.70643724698869, -24.61863174413091, -22.395372185205854, -22.465909692790895]
# n_states_5 = [-25.39271022169258, -25.134310266830813, -24.009501640247713, -23.593880476787557, -23.31005141154891]
# n_states_7 = [-23.99918504006178, -24.839157442748636, -26.45228727947142, -23.49605994363116, -23.990978644987013]
# n_states_4 = [-25.08335404524726, -24.504601023166725, -24.414900929208063, -26.589390944333175, -23.71076975934144]
# n_states_8 = [-25.367415386720435, -27.361826320066314, -23.831535480715278, -23.111846299018513]


# print(np.mean(n_states_4))
# print(np.mean(n_states_5))
# print(np.mean(n_states_6))
# print(np.mean(n_states_7))
# print(np.mean(n_states_8))

# print(np.isin(['a', 'b'], ['c,d']))
# test = set(np.array(['b','c']).tolist())
# s1 = [np.array(['a']), np.array(['b']), np.array(['b', 'c']), np.array(['a','b', 'c'])]
# row_condition = [set(i).issubset(test) for i in s1]
# # row_condition = [np.isin(i, test).any() and not(np.isin(i,test, invert = True).any()) for i in s1]

# print(row_condition)
# abc = [0].extend(['a', 'b'])
# print(abc)
# s2 = pd.Series(['c','d'])
# print(pd.Series(list(set(s1).intersection(set(s2)))))

# a = np.array([253952.0, 268952.0, 283952.0, 298952.0, 313952.0, 328952.0, 343952.0, 358952.0, 373952.0, 388952.0, 403952.0, 418952.0, 433952.0, 448952.0, 463952.0, 478952.0, 493952.0, 508952.0, 523952.0, 538952.0, 553952.0, 568952.0, 583952.0, 598952.0, 613952.0, 628952.0, 643952.0, 658952.0, 673952.0, 688952.0, 703952.0, 718952.0, 733952.0, 748952.0, 749904])
# print(np.diff(a))
# with open('./data/interim/20210726_new_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
#     task_to_storing_dict = pickle.load(f)

# subj_list = list(task_to_storing_dict['MPS'].keys())
# for subj in subj_list:
#     df_check = pd.DataFrame()
#     for task in tasks:
#         df_list = task_to_storing_dict[task][subj]
#         if task == "NMS" and subj == 35:
#             print(df_list)
#         for df_ in df_list:
#             df_check = pd.concat([df_check, df_])
#     df_check = df_check[['onset', 'offset', 'toy']]
#     df_check = df_check.sort_values(by=['onset'])
#     df_check.to_csv('./data/interim/check_reading_data/'+str(subj)+'.csv', index = False)

# with open("./data/interim/20210726_"+str(5)+"_no_ops_threshold_discretized_input_list_"+str(1.5)+"_min.pickle", 'rb') as f:
#     discretized_input_list_1 = pickle.load(f)

# with open("./data/interim/20210726_"+str(7)+"_no_ops_threshold_discretized_input_list_"+str(1.5)+"_min.pickle", 'rb') as f:
#     discretized_input_list_2 = pickle.load(f)

# with open('./data/interim/20210726_'+str(5)+'_no_ops_threshold_'+str(5)+'_states_prediction_'+str(1.5)+'_min.pickle', 'rb') as f:
#     pred_dict_1 = pickle.load(f)

# with open('./data/interim/20210726_'+str(7)+'_no_ops_threshold_'+str(5)+'_states_prediction_'+str(1.5)+'_min.pickle', 'rb') as f:
#     pred_dict_2 = pickle.load(f)

# with open('../data/interim/20210726_'+str(7)+'_no_ops_threshold_'+str(5)+'_states_prediction_'+str(1.5)+'_min.pickle', 'wb+') as f:
#     pred_dict_7s_1_5min = pickle.load(f)

# # print(np.equal(discretized_input_list_1,discretized_input_list_2))
# print(pred_dict_7s_1_5min)
# print(pred_dict_2)

# print(int("01"))

print('no_ops' in 'no_ops')
with open('./data/interim/20210726_7_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    clean_data_dict_og = pickle.load(f)

# with open("./data/interim/20210729_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min_2.pickle", 'rb') as f:
#     feature_dict = pickle.load(f)

# with open("./data/interim/20210729_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min_2.pickle", 'rb') as f:
#     time_arr_dict = pickle.load(f)

# with open("./data/interim/20210729_"+str(no_ops_threshold)+"_no_ops_threshold_label_"+str(interval_length)+"_min_2.pickle", 'rb') as f:
#     labels_dict = pickle.load(f)

with open('data/interim/20210729_merged_container_only_clean_data_for_feature_engineering.pickle', 'rb') as f:
    clean_data_dict = pickle.load(f)


subj = 7
task = "MPM"
og_df = pd.DataFrame()
for df_ in clean_data_dict_og[task][subj]:
    og_df = pd.concat([og_df, df_])

df_list = clean_data_dict[task][subj]
df = pd.DataFrame()
for df_ in df_list:
    df = pd.concat([df, df_])

onset = 330718
offset = 696592

title = "Subject 7, with caregiver, gross motor toys, original approach (top), revised approach (bottom)"

draw_comparison(subj, original_df = og_df, new_df = df, title = title, roi_onset= onset, roi_offset=offset)

#%%
def rank_state(model):
    # label_name_list = []
    # n_toy_list = []
    state_label_n_toy_dict = {}
    for idx, s in enumerate(model.states):
        if s.name == "no_toys":
            # for p in s.distribution.parameters[0]:
                # p.frozen = True
            state_label_n_toy_dict[idx] = 0
        # else: s.name:
            # state_label_n_toy_dict[idx] =
# %%
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
# model_file_name = "model_20210824_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'

model_file_path = Path('../models/hmm/20210824__30s_offset/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
# %%
# model.states[0].distribution.parameters[0][1]
model.states[0].distribution.parameters[0][2]

# %%
def rank_state_prev(model):
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
                state_label_n_toy_dict[idx] = np.dot(np.array(list(s.distribution.parameters[0][1].parameters[0].values())),np.array(list(s.distribution.parameters[0][1].parameters[0].keys())).T)+\
                                              np.dot(np.array(list(s.distribution.parameters[0][2].parameters[0].values())),np.array(list(s.distribution.parameters[0][2].parameters[0].keys())).T)
                
    # print(state_label_n_toy_dict)
    ranked_dict = {k: v for k, v in sorted(state_label_n_toy_dict.items(), key=lambda item: item[1])}
    return {v: str(k) for k, v in enumerate(ranked_dict.keys())}
# %%
ranked_dict = rank_state(model)
prev_ranked_dict = rank_state_prev(model)
print(ranked_dict)
print(prev_ranked_dict)

# %%
for s in ranked_dict.keys():
    print(model.states[s].distribution.parameters[0][1].parameters[0].values())

#%%
a = np.array([[1,2,3],["1", "2", "3"],[4, 2, 1]]).astype('object')
b = np.where(a == 2, "3", a)
print(b)

# %%
feature_set = 'n_new_toy_ratio'
interval_length = 1.5
no_ops_time = 10
with open("../data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_feature_dict_with_"+feature_set+"_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict = pickle.load(f)
# %%
n_features = 4
shift_time_list = np.arange(0, interval_length, .25)

len_list = []

input_list = np.empty((0, n_features))
# print(feature_dict)
for task in tasks:
    for subj, shifted_df_dict in feature_dict[task].items():
        for shift_time, feature_vector in shifted_df_dict.items():
            # if feature_set == 'n_new_toy_ratio_and_fav_toy_till_now':
                # m, n, _ = feature_vector.shape
                # feature_vector = feature_vector.reshape((n, m))
            # print(feature_vector.shape)

            input_list = np.vstack((input_list, feature_vector))
            # len_list.append(len(feature_vector))
# %%
plt.scatter(input_list[:,1], input_list[:,2])
plt.show()
# %%
feature_set = 'n_new_toy_ratio'
interval_length = 1.5
no_ops_time = 10
with open("../data/interim/20210805_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict = pickle.load(f)
n_features = 4
shift_time_list = np.arange(0, interval_length, .25)

len_list = []

input_list_2 = np.empty((0, n_features))
# print(feature_dict)
for task in tasks:
    for subj, shifted_df_dict in feature_dict[task].items():
        for shift_time, feature_vector in shifted_df_dict.items():
            # if feature_set == 'n_new_toy_ratio_and_fav_toy_till_now':
                # m, n, _ = feature_vector.shape
                # feature_vector = feature_vector.reshape((n, m))
            # print(feature_vector.shape)

            input_list_2 = np.vstack((input_list_2, feature_vector))
            # len_list.append(len(feature_vector))

# %%
plt.style.use('seaborn')
plt.rcParams.update({'font.size': 22})

fig=plt.figure(facecolor='white', figsize=(8,8))
# plt.scatter(input_list[:,2], input_list[:,1])
plt.scatter(input_list[:,3], input_list[:,1])
plt.xlabel('feature: fav toy ratio', fontdict ={'fontsize': 18})
plt.ylabel('feature: # toys',fontdict ={'fontsize': 18})
plt.rc('xtick', labelsize= 18)    
plt.rc('ytick', labelsize= 18) 
plt.grid(False)
# plt.title('Scatterplot of features: # toys and # new toys/# toys played', fontdict ={'fontsize': 18})
plt.show()
from scipy.stats import pearsonr

pearsonr(input_list[:,1], input_list[:,2])
# %%
plt.style.use('seaborn')
plt.rcParams.update({'font.size': 22})

fig=plt.figure(facecolor='white', figsize=(8,8))
plt.scatter(input_list_2[:,1], input_list_2[:,2])
plt.ylabel('feature: # new toys', fontdict ={'fontsize': 18})
plt.xlabel('feature: # toys',fontdict ={'fontsize': 18})
plt.rc('xtick', labelsize= 18)    
plt.rc('ytick', labelsize= 18)
plt.grid(False)
plt.title('Scatterplot of features: # toys and # new toys', fontdict ={'fontsize': 18})
plt.show()
pearsonr(input_list_2[:,1], input_list_2[:,2])
# %%
from scipy.stats import pearsonr

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
# %%

feature_set = 'n_new_toy_ratio'
interval_length = 1.5
no_ops_time = 10
n_states = 5
model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210815/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)
print(state_name_dict)
# %%
trans_matrix = pd.DataFrame(np.round(model.dense_transition_matrix()[:n_states+1,:n_states],2))
# trans_matrix.columns = list(state_name_dict.values())
file_name = 'trans_matrix_'+str(n_states)+"_states_seed_"+str(1)+'_'+str(interval_length)+'_min.csv'
file_path = Path('/scratch/mom_no_mom/reports/20210815/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/state_'+str(n_states))

trans_matrix = trans_matrix.rename(state_name_dict, axis=1) 
index = state_name_dict
index[n_states] = 'init_prob'
trans_matrix = trans_matrix.rename(index, axis = 0) 
save_path = file_path / file_name
trans_matrix.to_csv(save_path)
print(trans_matrix)
# %%
from matplotlib.patches import Rectangle, Patch

def draw_plain_timeline_with_feature_discretization_to_check(k, df, state_list, time_list, state_name, gap_size, state_color_dict, prob_list, shift):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.replace({'no_ops': 'no_toy'})
    # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toy': 'slategrey'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])

    if shift == 0.5:
        font_size = 16
    else:
        font_size = 12

    height = len(toys)
    time_list = np.array(time_list)/60000 - begin_time
    time = time_list[0] - .25
    # while time < time_list[-1]:
    if len(time_list) > 1:
        for idx, _ in enumerate(time_list):
            highest_states = prob_list[idx].argsort()[-2:][::-1]
            text=str(highest_states[0]) +' '+ str(np.round(prob_list[idx][highest_states[0]], 2)) +\
                "\n"+ str(highest_states[1]) +' '+ str(np.round(prob_list[idx][highest_states[1]], 2)) 

            if idx == 0:
                ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, ec = 'black', fill = False))
                ax.annotate(text, (time_list[0] - gap_size, height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')

            elif time_list[idx] - time_list[idx-1] <= gap_size:
                ax.add_patch(Rectangle((time_list[idx-1], 0), gap_size, height, ec = 'black', fill = False))
                ax.annotate(text, (time_list[idx-1], height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')
    else:
        highest_states = prob_list[0].argsort()[-2:][::-1]
        text=str(highest_states[0]) +' '+ str(np.round(prob_list[0][highest_states[0]], 2)) +\
            "\n"+ str(highest_states[1]) +' '+ str(np.round(prob_list[0][highest_states[1]], 2)) 
        ax.add_patch(Rectangle((time_list[0]-gap_size, 0), gap_size, height, fc = state_color_dict[state_name[state_list[0]]], ec = 'black', fill = True, alpha = 0.3))
        ax.annotate(text, (time_list[0] - gap_size, height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')

    plt.title('Subject ' + str(k), fontsize = 24)
    plt.xlabel('Minutes', fontsize = 24)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 24)
    plt.grid(False)
    plt.xticks(fontsize = 24)
    plt.ylim(top = height + 2)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])
    legend_elements = []
    for state in np.unique(state_list):
        legend_elements.append(Patch(facecolor=state_color_dict[state_name[state]], edgecolor=state_color_dict[state_name[state]], label=state_name[state], fill = True, alpha = 0.5))
    ax.legend(handles=legend_elements, loc='upper right', fontsize = 16)
    
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    # plt.close()
# %%
feature_set = 'n_new_toy_ratio'
interval_length = 2
no_ops_time = 5
n_states = 5
# with open('../data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
#     merged_pred_dict_all = pickle.load(f)

# with open('../data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
#     merged_proba_dict_all = pickle.load(f)

with open('../data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    all_proba_dict = pickle.load(f)

with open('../data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    pred_dict = pickle.load(f)
with open("../data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
    time_arr_dict = pickle.load(f)

with open('../data/interim/20210805_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    task_to_storing_dict = pickle.load(f)

model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210815/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

state_name_dict = rank_state(model)
# # %%
# merged_pred_dict_all['MPS'][1]
# merged_proba_dict_all['MPS'][1]
# %%
import importlib
import sys
# import inputs #import the module here, so that it can be reloaded.
# importlib.reload(visualization)
# importlib.reload(sys.modules['visualization'])
# importlib.reload(sys.modules['merge'])
from merge import merge_segment_with_state_calculation_all, merge_toy_pred
shift = .25
subj_list = list(task_to_storing_dict['MPS'].keys())
shift_time_list = np.arange(0,interval_length, shift)

merged_pred_dict_all = {}
merged_proba_dict_all = {}
time_subj_dict_all = {}
all_prob_dict_all = {}
for task in tasks:
    print(task)
    merged_df_dict = task_to_storing_dict[task]
    time_arr_shift_dict = time_arr_dict[task]
    pred_subj_dict = pred_dict[task]
    prob_subj_dict = all_proba_dict[task]

    merged_pred_dict_all_task_specific, merged_proba_dict_all_task_specific, time_subj_dict_all_task_specific, all_prob = merge_segment_with_state_calculation_all(subj_list, shift_time_list, merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size = interval_length, n_states = n_states, shift_interval = 60000*shift)

    merged_pred_dict_all[task] = merged_pred_dict_all_task_specific
    merged_proba_dict_all[task] = merged_proba_dict_all_task_specific
    time_subj_dict_all[task] = time_subj_dict_all_task_specific
    all_prob_dict_all[task] = all_prob
# %% 
subj_list
# %%
subj = 30
task = 'NMS'
df = pd.DataFrame()
for df_ in task_to_storing_dict[task][subj]:
    df = pd.concat([df, df_])
pred_state_list= pred_dict[task][subj][1]
# state_name_list = [state_name_dict[s] for s in pred_state_list]
time_list = time_arr_dict[task][subj][1]
# fig_name = './figures/hmm/state_distribution_20210815/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states/merged/'+task+'/'+str(subj)+".png"
draw_plain_timeline_with_feature_discretization_to_check(k = str(subj), \
                                                        df = df,\
                                                        state_list = pred_state_list,\
                                                        time_list = time_list,\
                                                        state_name = state_name_dict,\
                                                        gap_size = shift,\
                                                        state_color_dict= state_color_dict,\
                                                        prob_list = all_proba_dict[task][subj][1], 
                                                        shift = shift)
# %%
pred_dict['NMS'][30]
time_arr_dict['NMS'][30]
# %%
time_list
# %%
for feature_set in ['n_new_toy_ratio', 'fav_toy_till_now', 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio', 'new_toy_play_time_ratio']:
    for no_ops_time in [5, 7, 10]:
        for interval_length in [1, 1.5, 2]:
            for n_states in range(4, 7):
                model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
                model_file_path = Path('../models/hmm/20210815/'+feature_set)/model_file_name
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)

                state_name_dict = rank_state(model)
                trans_matrix = pd.DataFrame(np.round(model.dense_transition_matrix()[:n_states+1,:n_states],2))

                trans_matrix = trans_matrix.rename(state_name_dict, axis=1) 
                index = state_name_dict
                index[n_states] = 'init_prob'
                trans_matrix = trans_matrix.rename(index, axis = 0) 
                file_path = Path('/scratch/mom_no_mom/reports/20210815/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/state_'+str(n_states))
                file_name = 'trans_matrix_'+str(n_states)+"_states_seed_"+str(1)+'_'+str(interval_length)+'_min.csv'
                save_path = file_path / file_name
                trans_matrix.to_csv(save_path)

# %%
import numpy as np 
a = np.array([0, 0.33, 0.67, 1])
new_toys_bin = [0, .25, .5, .75]
discretized_n_new_toys = np.digitize(a, new_toys_bin, right = False)
print(discretized_n_new_toys)

new_toys_bin = [0, .2, .5, .6, .8]
discretized_n_new_toys = np.digitize(a, new_toys_bin, right = False)
print(discretized_n_new_toys)

# %%
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
interval_length = 1.5
shift = .5
with open('../data/interim/20210815_30s_offset'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    all_proba_dict = pickle.load(f)

with open('../data/interim/20210815_30s_offset'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    pred_dict = pickle.load(f)
with open("../data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
    time_arr_dict = pickle.load(f)

with open('../data/interim/20210805_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    task_to_storing_dict = pickle.load(f)
from merge import merge_segment_with_state_calculation_all
merged_pred_dict_all = {}
merged_proba_dict_all = {}
time_subj_dict_all = {}
all_prob_dict_all = {}
subj_list = list(task_to_storing_dict['MPS'].keys())
shift_time_list = np.arange(0,interval_length, shift)

for task in tasks:
    print(task)
    merged_df_dict = task_to_storing_dict[task]
    time_arr_shift_dict = time_arr_dict[task]
    pred_subj_dict = pred_dict[task]
    prob_subj_dict = all_proba_dict[task]

    merged_pred_dict_all_task_specific, merged_proba_dict_all_task_specific, time_subj_dict_all_task_specific, all_prob = merge_segment_with_state_calculation_all(subj_list, shift_time_list, merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size = interval_length, n_states = n_states, shift_interval = 60000*shift)

    merged_pred_dict_all[task] = merged_pred_dict_all_task_specific
    merged_proba_dict_all[task] = merged_proba_dict_all_task_specific
    time_subj_dict_all[task] = time_subj_dict_all_task_specific
    all_prob_dict_all[task] = all_prob
model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210815_30s_offset/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

for subj in subj_list:
    for task in tasks:
        path = Path('../figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'_new_merge/'+str(n_states)+'_states/merged/'+task+'/')
        path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame()
        for df_ in task_to_storing_dict[task][subj]:
            df = pd.concat([df, df_])
        pred_state_list= merged_pred_dict_all[task][subj]
        state_name_list = [state_name_dict[s] for s in pred_state_list]
        time_list = time_subj_dict_all[task][subj]
        prob_list = all_prob_dict_all[task][subj]
        fig_name = '../figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'_new_merge/'+str(n_states)+'_states/merged/'+task+'/'+str(subj)+".png"
        draw_timeline_with_prob_to_check(k = str(subj) + "window size: " + str(interval_length) + " no ops threshold "+ str(no_ops_time), \
                                        df = df, state_list = state_name_list, time_list = time_list,\
                                        state_name = state_name_dict, fig_name= fig_name, gap_size = shift, state_color_dict= state_color_dict,\
                                        prob_list = prob_list, shift = shift)
                                        
# %%



# %%
### test mini-timeline

from all_visualization_20210824 import rank_state


interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
# model_file_name = "model_20210824_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'

model_file_path = Path('../models/hmm/20210824__30s_offset/'+feature_set)/model_file_name

with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
with open('../data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    pred_dict = pickle.load(f)

with open('../data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    all_proba_dict = pickle.load(f)

with open('../data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_pred_dict_all = pickle.load(f)

with open('../data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_proba_dict_all = pickle.load(f)

# with open('..data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
#     time_subj_dict_all = pickle.load(f)

with open('../data/interim/20210824_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    task_to_storing_dict = pickle.load(f)

with open('../data/interim/20210818_baby_info.pickle', 'rb') as f:
    infant_info = pickle.load(f)


# %%
for i in merged_pred_dict_all['MPS'].values():
    print(len(i))
# %%
# rank infants by walking experience
ranked_infant_by_walk_exp = {}
for infant_id, walking_exp in infant_info['walking_exp'].items():
    ranked_infant_by_walk_exp[infant_id] = walking_exp//30

ranked_infant_dict = {k: v for k, v in sorted(ranked_infant_by_walk_exp.items(), key=lambda item: item[1])}
# print(ranked_infant_dict)
for task in tasks:
    fig_name = '../figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states/'+"condensed_timelines_"+task+".png"
    state_color_dict_shades = {"0":'gainsboro',  "1":'maroon', "2":'salmon', "3":'royalblue', "4":'midnightblue',  "5":'midnightblue', "6":'midnightblue', "7":'blue'}
    fig, axs = plt.subplots(nrows = len(ranked_infant_dict.keys()), ncols = 1, figsize = (30, 15), sharex = True)
    plt.suptitle("Condensed state sequences, "+task+",\ninfants ranked by increasing walking experience", fontsize = 52)
    for ax_id, infant_id in enumerate(ranked_infant_dict.keys()):
        state_list = [state_name_dict[i] for i in merged_pred_dict_all[task][infant_id]]
        session_len = len(state_list) if len(state_list) <= 16 else 16 
        for i in range(session_len):
            axs[ax_id].add_patch(Rectangle((i, 0), 1, 5, ec = 'black', fc = state_color_dict_shades[state_list[i]],fill = True, alpha = 0.7))
            axs[ax_id].set_xticks(np.arange(0, 18, 2))
            axs[ax_id].set_xticklabels([str(x) for x in np.arange(0, 9, 1)])
            axs[ax_id].set_ylabel('')
            axs[ax_id].set_yticklabels('')
    
    legend_elements = []
    for state in range(n_states):
        legend_elements.append(Patch(facecolor=state_color_dict_shades[str(state)],\
                                     edgecolor=state_color_dict_shades[str(state)],alpha = 0.7,\
                                     label=str(state)))

    # axs[ax_id].legend(handles=legend_elements, loc = 1, bbox_to_anchor=(1.05, 1), fontsize = 36, ncol=n_states)
    axs[ax_id].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize = 48)
    axs[ax_id].set_xlabel("Minutes", fontsize = 48)
    # plt.tight_layout()
    plt.savefig(fig_name)

#%%
for ax_id, infant_id in enumerate(ranked_infant_dict.keys()):
    state_list = [state_name_dict[i] for i in merged_pred_dict_all[task][infant_id]]
    print(state_list)

# %%

with open('../data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_theshold_'+str(n_states)+'_states_toy_pred_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
    toy_pred_list = pickle.load(f)
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
mobile_df = pd.DataFrame()
mobile_dict_for_std = {}
for state in state_name_dict.values():
    mobile_dict_for_std[state] = {}
    for toy in mobile_toys_list:
        mobile_dict_for_std[state][toy] = []
subj_list = list(task_to_storing_dict['MPS'].keys())

for task in ['MPM', "NMM"]:
    for subj in subj_list:
        if subj in toy_pred_list[task].keys():
            df_ = toy_pred_list[task][subj].copy()
            df_ = df_.explode('toys') 
            df_['toys'] = df_['toys'].replace({'no_ops':'no_toy'})

            df_['duration'] = df_['offset'] - df_['onset'] 
            df_['pred'] = df_['pred'].replace(state_name_dict)
            subj_mobile_dict = (df_.groupby(['pred', 'toys'])['duration'].sum()/df_.groupby(['pred'])['duration'].sum()).to_dict()
            for state in state_name_dict.values():
                for toy in mobile_toys_list:
                    key = (state, toy)
                    if key in subj_mobile_dict.keys():
                        mobile_dict_for_std[state][toy].append(subj_mobile_dict[key])
            mobile_df = pd.concat([mobile_df,  toy_pred_list[task][subj]])
mobile_df = mobile_df.explode('toys') 
mobile_df['toys'] = mobile_df['toys'].replace({'no_ops':'no_toy'})
mobile_df['duration'] = mobile_df['offset'] - mobile_df['onset'] 
mobile_df['pred'] = mobile_df['pred'].replace(state_name_dict)
mobile_toy_to_pred_dict = (mobile_df.groupby(['pred', 'toys'])['duration'].sum()/mobile_df.groupby(['pred'])['duration'].sum()).to_dict()
mobile_toy_list = list(mobile_df['toys'].dropna().unique())
mobile_std = {}
mobile_median = {}
for state in state_name_dict.values():
    mobile_std[state] = {}
    mobile_median[state] = {}
    for toy in mobile_toy_list:
        key = (state, toy)
        # print(key)
        if key in mobile_toy_to_pred_dict.keys():
            mobile_median[key] = np.median(mobile_dict_for_std[state][toy])
            mobile_std[state][toy] = np.std(mobile_dict_for_std[state][toy])

            # mobile_std[state][toy] = np.abs(np.sum(np.array(mobile_dict_for_std[state][toy])-mobile_toy_to_pred_dict[key]))/len(mobile_dict_for_std[state][toy])
#%%
subj_mobile_dict = (df_.groupby(['pred', 'toys'])['duration'].sum()/df_.groupby(['pred'])['duration'].sum()).to_dict()
print(subj_mobile_dict)
