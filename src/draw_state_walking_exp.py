#%%
import pickle
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique 
import seaborn
from variables import tasks, state_color_dict, state_color_dict_shades
from pathlib import Path 
from all_visualization_20210824 import rank_state
import numpy as np

with open('../data/interim/20210818_baby_info.pickle', 'rb') as f:
    infant_info = pickle.load(f)

feature_set = 'n_new_toy_ratio'
no_ops_time = 10
n_states = 5
interval_length = 1.5

with open('../data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_pred_dict_all = pickle.load(f)

model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210824__30s_offset/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)
subj_list = tuple(merged_pred_dict_all["MPS"].keys())
# print(merged_pred_dict_all.keys())

#%%
print(infant_info['crawling_exp'].values())
plt.hist(infant_info['crawling_exp'].values())
plt.title("histogram of crawling exp")

# %% 
print(
# %%
print(infant_info['walking_exp'].values())

walk_exp_month = np.floor(np.array(list(infant_info['walking_exp'].values()))/30)
mnth, cnt = np.unique(walk_exp_month, return_counts = True)
# plt.hist(np.floor(np.array(list(infant_info['walking_exp'].values()))/30))
plt.bar(mnth, cnt)
plt.xticks(np.arange(0, 11), [str(x) for x in np.arange(0, 11)])
plt.yticks(np.arange(10), [str(x) for x in np.arange(10)])

plt.xlabel("walking experience in month(s)")
plt.ylabel("# infants")
plt.grid(False)
plt.title("histogram of walking exp")
# %%
all_task_dict = {}
all_task_time = {}

each_cond_cnt = {}
each_cond_time = {}

each_cond_time_2_mon = {}
all_task_time_2_mon = {}

each_cond_time_3_mon = {}
all_task_time_3_mon = {}

for state in state_name_dict.values():
    all_task_dict[state] = []
    all_task_time[state] = {}
    all_task_time_2_mon[state] = {}
    all_task_time_3_mon[state] = {}

    for i in range(1, 11):
        all_task_time[state][i] = []
        if i <= 6:
            all_task_time_2_mon[state][i] = []
            all_task_time_3_mon[state][i] = []


for task in tasks:
    each_cond_cnt[task] = {}
    each_cond_time[task] = {}
    each_cond_time_2_mon[task] = {}
    each_cond_time_3_mon[task] = {}

    for state in state_name_dict.values():
        each_cond_cnt[task][state] = []
        each_cond_time[task][state] = {}
        each_cond_time_2_mon[task][state] = {}
        each_cond_time_3_mon[task][state] = {}
        for i in range(1, 11):
            each_cond_time[task][state][i] = []
            if i <= 6:
                each_cond_time_2_mon[task][state][i] = []
            if i <= 3:
                each_cond_time_3_mon[task][state][i] = []

    
bins = [0, 30, 60, 90, 120, 150, 180, 240, 270, 300]
bins_2 = np.arange(0, 300, 60)
bins_3 = np.arange(0, 181, 90)

# print(np.amax(infant_info['walking_exp'].values()))
# print(each_cond_time)
for task in tasks:
    for subj in subj_list:
        pred = [state_name_dict[s] for s in merged_pred_dict_all[task][subj]]
        pred_unique, cnt = np.unique(pred, return_counts= True)
        # print(pred_unique, cnt)
        walking_exp = infant_info['walking_exp'][subj]
        walk_exp_month_2_mon = np.digitize(walking_exp, bins = bins_2, right = False)
        walk_exp_month = np.digitize(walking_exp, bins = bins, right = False)
        walking_exp_3_mon = np.digitize(walking_exp, bins = bins_3, right = False)
        for state in range(n_states):
            if not str(state) in pred_unique:
                pred_unique = np.append(pred_unique, str(state))
                cnt = np.append(cnt, 0)

        # walk_exp_month = walking_exp//30+1
        if walk_exp_month > 10:
            walk_exp_month = 10
        for idx, state in enumerate(pred_unique):
            # print(walking_exp, walk_exp_month)
            each_cond_time[task][str(state)][walk_exp_month].append(cnt[idx]/cnt.sum())
            each_cond_time_2_mon[task][state][walk_exp_month_2_mon].append(cnt[idx]/cnt.sum())
            each_cond_time_3_mon[task][state][walking_exp_3_mon].append(cnt[idx]/cnt.sum())

            
            each_cond_cnt[task][state].append(walking_exp)
            all_task_time[str(state)][walk_exp_month].append(cnt[idx]/cnt.sum()) 
            all_task_time_2_mon[str(state)][walk_exp_month_2_mon].append(cnt[idx]/cnt.sum()) 
            all_task_time_3_mon[str(state)][walking_exp_3_mon].append(cnt[idx]/cnt.sum()) 

            all_task_dict[str(state)].append(walking_exp)
#%%
each_cond_time_3_mon["MPS"]
# %%
# print(all_task_time['1'])
# %%
# plt.figure(figsize = (10, 8))
fig, axs = plt.subplots(nrows= 1, ncols= len(state_name_dict.values()), figsize = (20, 10), sharey = True)
plt.suptitle("Avg. time spent in each state for each walking exp. group", fontsize = 26)

for idx, state in enumerate(state_name_dict.values()):
    # print(all_task_time[state])

    for month_pos, val in all_task_time[state].items():
        if len(val) == 0:
            axs[idx].bar(month_pos, 0)
        else:
            axs[idx].bar(month_pos, np.mean(val), color = state_color_dict[state])
            axs[idx].errorbar(month_pos,np.mean(val), yerr = ([0], [np.std(val)]), barsabove = True, color = 'dimgray')

    axs[idx].set_xlabel('state ' + state, fontsize=24)
    axs[idx].xaxis.set_label_position('top')
    axs[idx].set_xticks(list(range(1, 11)))
    axs[idx].set_xticklabels([str(i) for i in range(1, 11)], fontsize=20)
for ax in axs:
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.grid(False)
# plt.xlabel('Walking exp in days')
axs[0].set_ylabel("% in the session", fontsize=24)
axs[0].set_yticks(np.arange(0, 0.9, 0.1))
axs[0].set_yticklabels([str(np.round(i, 1)) for i in np.arange(0, 0.9, 0.1)], fontsize = 24)
fig.text(0.5, -0.01, 'Walking exp in months', ha='center', fontsize=24)
plt.tight_layout()
plt.show()

# %%
# plt.figure(figsize = (10, 8))
fig, axs = plt.subplots(nrows= 1, ncols= len(state_name_dict.values()), figsize = (20, 10), sharey = True)
plt.suptitle("Distribution of walking experience (days) in each state", fontsize = 18)

for idx, state in enumerate(state_name_dict.values()):
    axs[idx].hist(all_task_dict[state])
    axs[idx].set_xlabel('state ' + state)
    axs[idx].xaxis.set_label_position('top')

    # axs[idx].set_xticklabels([str(i) for i in range(1, 11)])
    # axs[idx].set_xticklabels([str(i) for i in range(1, 11)])
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.grid(False)
# plt.xlabel('Walking exp in days')
axs[0].set_ylabel('# of infants')
fig.text(0.5, 0.04, 'Walking exp in month(s)', ha='center', fontsize=16)
plt.show()

# %%
# plot each row is a condition (MPS, MPM, NMS, NMM), each col is a state, each cell is the pct. spent in each state for each group
fig, axs = plt.subplots(nrows= 4, ncols= len(state_name_dict.values()), figsize = (20, 15), sharex = True, sharey = True)
plt.suptitle("Pct. of each session in each state for each walking exp. group", fontsize = 32)

for row_idx, task in enumerate(tasks):
    for col_idx, state in enumerate(state_name_dict.values()):
        for month_pos, all_pct in each_cond_time_2_mon[task][str(state)].items():
            if len(all_pct) == 0:
                axs[row_idx, col_idx].bar(month_pos, 0)
            else:
                axs[row_idx, col_idx].bar(month_pos, np.mean(all_pct), color = state_color_dict[state])
                axs[row_idx, col_idx].errorbar(month_pos, np.mean(all_pct), yerr = ([0],[np.std(all_pct)]), barsabove = True, color = 'dimgray')
            axs[row_idx, col_idx].grid(False)
        axs[row_idx, col_idx].set_xticks(list(each_cond_time_2_mon[task][str(state)].keys()))
        axs[row_idx, col_idx].set_xticklabels(np.arange(2, 14,2), fontsize = 28)
        # axs[row_idx, -1].set_xticks(np.arange(1, 7))
        # axs[row_idx, -1].set_xticklabels(np.arange(2, 14,), fontsize = 18)
        if row_idx == 0:
            axs[row_idx, col_idx].set_xlabel("state " + state, fontsize  = 28)
            axs[row_idx, col_idx].xaxis.set_label_position('top')
        if row_idx == len(tasks):
            axs[row_idx, 0].tick_params("y", left=False, labelleft=False)
            axs[row_idx, -1].tick_params("y", right=True, labelright=True)
        if col_idx == 0:
            axs[row_idx, col_idx].set_ylabel(task, fontsize = 28)
            axs[row_idx, col_idx].set_yticks([0, 0.2, .4, .6, .8])
            axs[row_idx, col_idx].set_yticklabels([0, 0.2, .4, .6, .8], fontsize = 28)

fig.text(0.5, -0.01, 'Walking exp. (in months)', ha='center', fontsize = 28)
fig.text(-0.01, 0.5, '% time in a session', va='center', rotation='vertical', fontsize = 28)

plt.tight_layout()
plt.show()

# %%
# plot each row is a condition (MPS, MPM, NMS, NMM), each col is a walking exp. group, distribution of time they spend in the state

fig, axs = plt.subplots(nrows= 4, ncols= 6, figsize = (20, 15), sharex = True, sharey = True)
plt.suptitle("Pct. of each session in each state for each walking exp. group", fontsize = 28)
# tasks = ['MPS', 'NMS', 'MPM', 'NMM']
for row_idx, task in enumerate(tasks):
    for state_pos, state in enumerate(state_name_dict.values()):
        for col_idx, (month_pos, all_pct) in enumerate(each_cond_time_2_mon[task][state].items()):
            if len(all_pct) == 0:
                axs[row_idx, col_idx].bar(state_pos, 0)
            else:
                axs[row_idx, col_idx].bar(state_pos, np.mean(all_pct), color = state_color_dict_shades[state])
                axs[row_idx, col_idx].errorbar(state_pos, np.mean(all_pct), yerr = ([0],[np.std(all_pct)]), barsabove = True, color = 'dimgray')
            axs[row_idx, col_idx].grid(False)
            axs[row_idx, col_idx].set_xticks(list(range(len(state_name_dict.values()))))
            axs[row_idx, col_idx].set_xticklabels(state_name_dict.values(), fontsize = 26)
            
            if row_idx == 0:
                axs[row_idx, col_idx].set_xlabel(str(month_pos*2) + " month(s)", fontsize  = 26)
                axs[row_idx, col_idx].xaxis.set_label_position('top')
            if row_idx == len(tasks):
                axs[row_idx, 0].tick_params("y", left=False, labelleft=False)
                axs[row_idx, -1].tick_params("y", right=True, labelright=True)
                

                
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(task, fontsize = 26)
                axs[row_idx, col_idx].set_yticks([0, 0.2, .4, .6, .8])
                axs[row_idx, col_idx].set_yticklabels([0, 0.2, .4, .6, .8], fontsize = 26)
fig.text(0.5, -0.01, 'States', ha='center', fontsize = 26)
fig.text(-0.01, 0.5, '% time in a session', va='center', rotation='vertical', fontsize = 26)

plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(nrows= 4, ncols= 3, figsize = (12, 15), sharex = False, sharey = True)
plt.suptitle("Pct. of each session in each state for each walking exp. group", fontsize = 28)
xlabels_list = ['0-3 month(s)', '3-6 months', '6+ months'] 
# tasks = ['MPS', 'NMS', 'MPM', 'NMM']
for row_idx, task in enumerate(tasks):
    for state_pos, state in enumerate(state_name_dict.values()):
        for col_idx, (month_pos, all_pct) in enumerate(each_cond_time_3_mon[task][state].items()):
            if len(all_pct) == 0:
                axs[row_idx, col_idx].bar(state_pos, 0)
            else:
                axs[row_idx, col_idx].bar(state_pos, np.mean(all_pct), color = state_color_dict_shades[state], alpha = 0.8)
                axs[row_idx, col_idx].errorbar(state_pos, np.mean(all_pct), yerr = ([0],[np.std(all_pct)]), barsabove = True, color = 'dimgray')
            axs[row_idx, col_idx].grid(False)
            axs[row_idx, col_idx].set_xticks(list(range(len(state_name_dict.values()))))
            axs[row_idx, col_idx].set_xticklabels(state_name_dict.values(), fontsize = 26)
            
            # if row_idx == 0:
            axs[row_idx, col_idx].set_xlabel(xlabels_list[col_idx], fontsize  = 26)
            axs[row_idx, col_idx].xaxis.set_label_position('top')
            # if row_idx == len(tasks):
            #     axs[row_idx, 0].tick_params("y", left=False, labelleft=False)
            #     axs[row_idx, -1].tick_params("y", right=True, labelright=True)

                
            if col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(task, fontsize = 26)
                axs[row_idx, col_idx].set_yticks([0, 0.2, .4, .6, .8])
                axs[row_idx, col_idx].set_yticklabels([0, 0.2, .4, .6, .8], fontsize = 26)
fig.text(0.5, -0.01, 'States', ha='center', fontsize = 26)
fig.text(-0.01, 0.5, '% time in a session', va='center', rotation='vertical', fontsize = 26)

plt.tight_layout()
plt.show()
# %%
bins = [0, 30, 60, 90, 120, 150, 180, 240, 270, 300]
# plot each col is a state, each cell is the pct. time spent in each state for each group
fig, axs = plt.subplots(nrows= 1, ncols= len(state_name_dict.values()), figsize = (20, 10), sharey = True, sharex = True)
plt.suptitle("Distribution of walking experience (months) in each state", fontsize = 32)

for idx, state in enumerate(state_name_dict.values()):
    converted_to_month = np.digitize(all_task_dict[state], bins = bins)
    unique_months, cnt = np.unique(converted_to_month, return_counts=True)
    print(unique_months)
    cnt = cnt/cnt.sum()
    axs[idx].bar(unique_months, cnt, color = state_color_dict[state])
    axs[idx].set_xlabel('state ' + state, fontsize=28)
    axs[idx].xaxis.set_label_position('top')
    axs[idx].set_xticks(range(1, 11))
    axs[idx].set_xticklabels([str(i) for i in range(1, 11)], fontsize=28)
    axs[idx].set_yticks(np.arange(0, 0.31, 0.05))
    axs[idx].set_yticklabels([str(np.round(i, 2)) for i in np.arange(0, 0.31, 0.05)], fontsize=28)

    # axs[idx].set_xticklabels([str(i) for i in range(1, 11)])
for ax in axs:
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.grid(False)
# plt.xlabel('Walking exp in days')
axs[0].set_ylabel("% of infants pop. in each state", fontsize=28)
fig.text(0.5, -0.01, 'Walking exp in month(s)', ha='center', fontsize=28)
plt.tight_layout()
plt.show()


# %%
# plot each col is a state, each row is a condition, each cell is the walking exp. dist in that state-condition
bins = [0, 30, 60, 90, 120, 150, 180, 240, 270, 300]
fig, axs = plt.subplots(nrows= 4, ncols= len(state_name_dict.values()), figsize = (20, 15), sharex = True, sharey = True)
plt.suptitle("Distribution of walking experience in each state", fontsize = 32)

for row_idx, task in enumerate(each_cond_cnt.keys()):
    for col_idx, state in enumerate(each_cond_cnt[task].keys()):
        converted_to_month = np.digitize(each_cond_cnt[task][state], bins = bins)
        unique_months, cnt = np.unique(converted_to_month, return_counts=True)
        cnt = cnt/cnt.sum()
        axs[row_idx, col_idx].bar(unique_months, cnt, color = state_color_dict[state])
        axs[row_idx, col_idx].grid(False)   
        axs[row_idx, col_idx].set_xticks(list(range(1, 11)))
        axs[row_idx, col_idx].set_xticklabels([str(i) for i in range(1, 11)], fontsize = 28)
        if row_idx == len(tasks):
            axs[row_idx, 0].tick_params("y", left=False, labelleft=False)
            axs[row_idx, -1].tick_params("y", right=True, labelright=True)
            
        if col_idx == 0:
            axs[row_idx, col_idx].set_ylabel(task, fontsize = 28)
            axs[row_idx, col_idx].set_yticks([0, 0.2, .4, .6, .8, 1])
            axs[row_idx, col_idx].set_yticklabels([0, 0.2, .4, .6, .8, 1], fontsize = 28)
        if row_idx == 0:
            axs[row_idx, col_idx].set_xlabel("state " + state, fontsize  = 28)
            axs[row_idx, col_idx].xaxis.set_label_position('top')

fig.text(0.5, -0.01, 'Walking exp. (in months)', ha='center', fontsize = 28)
fig.text(-0.01, 0.5, 'Distribution of walking exp.', va='center', rotation='vertical', fontsize = 28)

plt.tight_layout()
plt.show()
# %%
