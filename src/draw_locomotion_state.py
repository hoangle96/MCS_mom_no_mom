# %%
import numpy as np
from all_visualization_20210824 import rank_state
from variables import tasks
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.lines import Line2D
# %%
n_states = 5
feature_set = "n_new_toy_ratio"
no_ops_time = 10
interval_length = 1.5
with open('../data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_locomotion_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_pred_w_locomotion = pickle.load(f)

# %%
model_file_name = "model_20210907_"+feature_set+"_" + \
    str(interval_length)+"_interval_length_"+str(no_ops_time) + \
    "_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)
# %%
# ## toy locomotion figure
movement_time_by_each_task = {}
steps_by_each_task = {}

movement_time_by_each_task_for_std = {}
steps_by_each_task_for_std = {}

movement_time_by_each_task_mean_each_infant = {}
steps_by_each_task_mean_each_infant = {}

for task in tasks:
    if task not in movement_time_by_each_task.keys():
        movement_time_by_each_task[task] = {}
        steps_by_each_task[task] = {}

        steps_by_each_task_for_std[task] = {}
        movement_time_by_each_task_for_std[task] = {}

        movement_time_by_each_task_mean_each_infant[task] = {}
        steps_by_each_task_mean_each_infant[task] = {}

        for state in range(n_states):
            movement_time_by_each_task[task][state] = []
            steps_by_each_task[task][state] = []

            movement_time_by_each_task_mean_each_infant[task][state] = []
            steps_by_each_task_mean_each_infant[task][state] = []

    for subj, df in merged_pred_w_locomotion[task].items():
        df['pred'] = df['pred'].replace(state_name_dict)
        # print(df['pred'].dtype)
        # print(df.head())
        for state in range(n_states):
            # print(state)
            df_ = df.loc[df.loc[:, 'pred'] == str(state), :]
            # print(df_.head())

            if len(df_) > 0:
                steps = df_['steps'].to_numpy()*2
                movement_time = df_['movement_time'].to_numpy()/30000
                steps_mean = np.mean(df_['steps'].to_numpy()*2)
                movement_time_mean = np.mean(
                    df_['movement_time'].to_numpy()/30000)
            else:
                steps = []
                movement_time = []
                steps_mean = 0
                movement_time_mean = 0
                steps_by_each_task_for_std[task][state] = None
                movement_time_by_each_task_for_std[task][state] = None
            steps_by_each_task[task][state].extend(steps)
            movement_time_by_each_task[task][state].extend(movement_time)

            movement_time_by_each_task_mean_each_infant[task][state].append(
                movement_time_mean)
            steps_by_each_task_mean_each_infant[task][state].append(steps_mean)

    for state in range(n_states):
        steps_by_each_task_for_std[task][state] = np.sqrt(np.mean(np.abs(
            movement_time_by_each_task_mean_each_infant[task][state]-np.mean(steps_by_each_task[task][state]))**2))
        movement_time_by_each_task_for_std[task][state] = np.sqrt(np.mean(np.abs(
            movement_time_by_each_task_mean_each_infant[task][state]-np.mean(movement_time_by_each_task[task][state]))**2))

fig_path = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_' + \
    str(no_ops_time)+'/window_size_'+str(interval_length)+'/' + \
    str(n_states)+"_states/step_by_state_with_std_mean.png"

# %%
mean_dict['MPS'][1]

# %%
ylabel = 'avg # steps/min'
title = "Avg number of steps in each state for each condition"
task_list = ["MPS", "NMS", "MPM", "NMM"]
mean_dict = steps_by_each_task
plt.style.use('seaborn')
offset = 20
fig, ax = plt.subplots(figsize=(30, 16))
condition_name = {
    'MPS': "With caregivers, fine motor toys",
    'MPM': "With caregivers, gross motor toys",
    'NMS': "Without caregivers, fine motor toys",
    'NMM': "Without caregivers, gross motor toys",
}
task_edge_color = {"MPS": 'r', 'MPM': 'b', "NMS": 'r', 'NMM': 'b'}
task_face_color = {"MPS": 'r', 'MPM': 'b', "NMS": 'none', 'NMM': 'none'}
task_linestyle = {"MPS": '-', 'MPM': '-', "NMS": "--", 'NMM': "--"}
median_color = {"MPS": 'w', 'MPM': 'w', "NMS": 'r', 'NMM': 'b'}
fill_style_dict = {"MPS": 'full', 'MPM': 'full', "NMS": 'none', 'NMM': 'none'}
for state_position in range(n_states):
    if state_position == 1:
        for task_idx, task in enumerate(task_list):
            data = mean_dict[task][state_position]
            pos = state_position*offset + task_idx*4
            # print(task, pos)
#
            plt.boxplot(x=mean_dict[task][state_position], positions=[state_position*offset + task_idx*4],
                        showfliers=True, widths=3, patch_artist=True,
                        boxprops=dict(
                            facecolor=task_face_color[task], edgecolor=task_edge_color[task]),
                        capprops=dict(
                            c=task_edge_color[task], ls=task_linestyle[task], lw=3),
                        whiskerprops=dict(
                            c=task_edge_color[task], ls=task_linestyle[task], lw=3),
                        flierprops=dict(
                            markerfacecolor=task_face_color[task], markeredgecolor=task_edge_color[task]),
                        medianprops=dict(
                            c=median_color[task], ls=task_linestyle[task], lw=3),
                        meanprops=dict(c=median_color[task], ls=task_linestyle[task], lw=3))
            # print(val)
    else:
        for task_idx, task in enumerate(task_list):
            if len(mean_dict[task][state_position]) == 0:
                data = [0]
            else:
                data = mean_dict[task][state_position]
            plt.boxplot(x=data, positions=[state_position*offset + task_idx*4], showfliers=True,
                        widths=3, patch_artist=True,
                        boxprops=dict(
                            facecolor=task_face_color[task], edgecolor=task_edge_color[task]),
                        capprops=dict(
                            c=task_edge_color[task], ls=task_linestyle[task], lw=3),
                        whiskerprops=dict(
                            c=task_edge_color[task], ls=task_linestyle[task], lw=3),
                        flierprops=dict(
                            markerfacecolor=task_face_color[task], markeredgecolor=task_edge_color[task]),
                        medianprops=dict(c=median_color[task], ls=task_linestyle[task], lw=3))

    if state_position != n_states - 1:
        ax.axvline(state_position*offset + offset //
                   2 + 4, color='grey', alpha=0.1)

handles = []
for task in task_list:
    handles.append(Line2D([0], [0], marker='o', c=task_edge_color[task], label=condition_name[task],
                          markerfacecolor=task_face_color[task], markersize=15, ls=task_linestyle[task],
                          fillstyle=fill_style_dict[task]))
plt.grid(False)
ax.xaxis.grid(b=True, which="minor", color='grey',
              linestyle='--', linewidth=1, alpha=.5)
ax.legend(loc=2, fontsize=24, handles=handles)
ax.set_xticks(np.arange(5.5, 5.5+offset*n_states, offset))
ax.set_xticklabels([str(i) for i in range(n_states)], fontsize=28)
# ax.set_yticklabels()
ax.set_xlabel("States", fontsize=28)
ax.set_ylabel(ylabel, fontsize=28)
plt.yticks(fontsize=28)

ax.set_facecolor('white')
# ax.set_ylim(bottom =0)
# plt.tight_layout()
plt.title(title, fontsize=32)

plt.show()
# plt.savefig(figname)
plt.close()

# %%
plt.boxplot(x=mean_dict['MPS'][1])
