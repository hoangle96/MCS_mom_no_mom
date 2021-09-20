# %%
from statsmodels.stats.weightstats import ztest
import numpy as np
import pickle
from pathlib import Path
from all_visualization_20210824 import rank_state
from variables import tasks
# ztest = CompareMeans.ztest_ind


# %%
del ztest

# %%
# Test the difference of mean of locomotion in different states
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
with open('../data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_locomotion_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_pred_w_locomotion = pickle.load(f)


model_file_name = "model_20210907_"+feature_set+"_" + \
    str(interval_length)+"_interval_length_"+str(no_ops_time) + \
    "_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

# %%
movement_time_by_each_task = {}
steps_by_each_task = {}

movement_time_by_each_state = {}
steps_by_each_state = {}

for task in tasks:
    if task not in movement_time_by_each_task.keys():

        movement_time_by_each_task[task] = {}
        steps_by_each_task[task] = {}

        for state in range(n_states):
            movement_time_by_each_task[task][state] = []
            steps_by_each_task[task][state] = []

            movement_time_by_each_state[state] = []
            steps_by_each_state[state] = []

    for subj, df in merged_pred_w_locomotion[task].items():
        # movement_time_by_state[task].append()
        df['pred'] = df['pred'].replace(state_name_dict)
        for state in range(n_states):
            # print(state)
            df_ = df.loc[df.loc[:, 'pred'] == str(state), :]
            if len(df_) > 0:
                steps_mean = np.mean(df_['steps'].to_numpy()*2)
                movement_time_mean = np.mean(
                    df_['movement_time'].to_numpy()/30000)
            else:
                steps_mean = 0
                movement_time_mean = 0

            # print(df.loc[df.loc[:,'pred'] == str(state), 'steps'].to_numpy()*4)
            # steps = np.nanmean(df.loc[df.loc[:,'pred'] == str(state), 'steps'].to_numpy()*4)

            # movement_time = np.nanmean(df.loc[df.loc[:,'pred'] == str(state), 'movement_time'].to_numpy()/15000)

            steps_by_each_task[task][state].append(steps_mean)
            movement_time_by_each_task[task][state].append(movement_time_mean)
# %%
steps_by_each_task
# %%
steps_by_each_task["MPS"]
# %%
steps_by_each_task
# %%
print("testing mobile v stationary each state")
for state in range(n_states):
    print("state: ", state)
    print("step cnt avg")
    print(ztest(steps_by_each_task['NMM'][state]+steps_by_each_task['MPM'][state],
                steps_by_each_task['NMS'][state]+steps_by_each_task['MPS'][state], alternative='larger'))
    print("movement time avg")
    print(ztest(movement_time_by_each_task['NMM'][state]+movement_time_by_each_task['MPM']
                [state], movement_time_by_each_task['NMS'][state]+movement_time_by_each_task['MPS'][state]))

# %%
print("step cnt, testing caregiver v no-caregiver")
for state in range(n_states):
    print('\n')
    print("state: ", state)
    print('gross motor toy')
    print(ztest(steps_by_each_task['NMM'][state],
                steps_by_each_task['MPM'][state], alternative='larger'))
    print('fine motor toy')
    print(ztest(steps_by_each_task['NMS'][state],
                steps_by_each_task['MPS'][state], alternative='larger'))
# %%
print("movement, testing caregiver v no-caregiver")
for state in range(n_states):
    print('\n')
    print('gross motor toy')
    print(ztest(movement_time_by_each_task['NMM'][state],
                movement_time_by_each_task['MPM'][state], alternative='larger'))
    print('fine motor toy')
    print(ztest(movement_time_by_each_task['NMS'][state],
                movement_time_by_each_task['MPS'][state], alternative='larger'))

# %%
# Compare across states
print("test movement time")
print("Fine motor toy, no caregiver, state 3 > 2", ztest(
    movement_time_by_each_task['NMS'][3], movement_time_by_each_task['NMS'][2], alternative='larger'))
print("Fine motor toy, no caregiver, state 3 > 4", ztest(
    movement_time_by_each_task['NMS'][3], movement_time_by_each_task['NMS'][4], alternative='larger'))
print("Fine motor toy, no caregiver, state 2 > 1", ztest(
    movement_time_by_each_task['NMS'][2], movement_time_by_each_task['NMS'][1], alternative='larger'))

print("Fine motor toy, with caregiver, state 3 > 2", ztest(
    movement_time_by_each_task['MPS'][3], movement_time_by_each_task['MPS'][2], alternative='larger'))
print("Fine motor toy, with caregiver, state 3 > 1", ztest(
    movement_time_by_each_task['MPS'][3], movement_time_by_each_task['MPS'][1], alternative='larger'))
print("Fine motor toy, with caregiver, state 3 > 4", ztest(
    movement_time_by_each_task['MPS'][3], movement_time_by_each_task['MPS'][4], alternative='larger'))
print("Fine motor toy, with caregiver, state 2 > 1", ztest(
    movement_time_by_each_task['MPS'][2], movement_time_by_each_task['MPS'][1], alternative='larger'))
print("\n")
print("Gross motor toy, no caregiver, state 3 > 4", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][4], alternative='larger'))
print("Gross motor toy, no caregiver, state 3 > 2", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][2], alternative='larger'))
print("Gross motor toy, no caregiver, state 3 > 1", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][1], alternative='larger'))
print("Gross motor toy, with caregiver, state 3 > 4", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][4], alternative='larger'))
print("Gross motor toy, with caregiver, state 3 > 2", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][2], alternative='larger'))
print("Gross motor toy, with caregiver, state 3 > 1", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][1], alternative='larger'))

print("Gross motor toy, with caregiver, state 3 > 1", ztest(
    movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][1], alternative='larger'))

# %%
print("Gross motor toy, E states > F states",\
      ztest(movement_time_by_each_task['NMM'][3] + movement_time_by_each_task['NMM'][4] + movement_time_by_each_task['MPM'][3] + movement_time_by_each_task['MPM'][4],\
            movement_time_by_each_task['NMM'][1] + movement_time_by_each_task['NMM'][2] + movement_time_by_each_task['MPM'][1] + movement_time_by_each_task['MPM'][2],\
            alternative='larger'))
print("Fine motor toy, E states > F states",\
      ztest(movement_time_by_each_task['NMS'][3] + movement_time_by_each_task['NMS'][4] + movement_time_by_each_task['MPS'][3] + movement_time_by_each_task['MPS'][4],\
            movement_time_by_each_task['NMS'][1] + movement_time_by_each_task['NMS'][2] + movement_time_by_each_task['MPS'][1] + movement_time_by_each_task['MPS'][2],\
            alternative='larger'))
# print("Gross motor toy, no caregiver, state 3 > 4", ztest(
#     movement_time_by_each_task['NMM'][3], movement_time_by_each_task['NMM'][4], alternative='larger'))



# %%
print("test no. steps ")
print("Fine motor toy, no caregiver, state 3 > 2", ztest(
    steps_by_each_task['NMS'][3], steps_by_each_task['NMS'][2], alternative='larger'))
print("Fine motor toy, no caregiver, state 3 > 4", ztest(
    steps_by_each_task['NMS'][3], steps_by_each_task['NMS'][4], alternative='larger'))
print("Fine motor toy, no caregiver, state 2 > 1", ztest(
    steps_by_each_task['NMS'][2], steps_by_each_task['NMS'][1], alternative='larger'))
print("Fine motor toy, no caregiver, state 1 > 0", ztest(
    steps_by_each_task['NMS'][1], steps_by_each_task['NMS'][0], alternative='larger'))
print("Fine motor toy, no caregiver, state 2 > 0", ztest(
    steps_by_each_task['NMS'][2], steps_by_each_task['NMS'][0], alternative='larger'))
print("Fine motor toy, no caregiver, state 3 > 0", ztest(
    steps_by_each_task['NMS'][3], steps_by_each_task['NMS'][0], alternative='larger'))
print("Fine motor toy, no caregiver, state 4 > 0", ztest(
    steps_by_each_task['NMS'][4], steps_by_each_task['NMS'][0], alternative='larger'))

print("Fine motor toy, with caregiver, state 3 > 2", ztest(
    steps_by_each_task['MPS'][3], steps_by_each_task['MPS'][2], alternative='larger'))
print("Fine motor toy, with caregiver, state 3 > 1", ztest(
    steps_by_each_task['MPS'][3], steps_by_each_task['MPS'][1], alternative='larger'))
print("Fine motor toy, with caregiver, state 3 > 4", ztest(
    steps_by_each_task['MPS'][3], steps_by_each_task['MPS'][4], alternative='larger'))
print("Fine motor toy, with caregiver, state 2 > 1", ztest(
    steps_by_each_task['MPS'][2], steps_by_each_task['MPS'][1], alternative='larger'))
print("Fine motor toy, state 3 > 2", ztest(steps_by_each_task['NMS'][3] + steps_by_each_task['MPS'][3] + steps_by_each_task['NMS'][3] + steps_by_each_task['MPS'][3],
                                           steps_by_each_task['NMS'][2] + steps_by_each_task['MPS'][2] +
                                           steps_by_each_task['NMS'][2] +
                                           steps_by_each_task['MPS'][2],
                                           alternative='larger'))
print("Fine motor toy, with caregiver, state 1 > 0", ztest(
    steps_by_each_task['MPS'][1], steps_by_each_task['MPS'][0], alternative='larger'))
print("Fine motor toy, with caregiver, state 2 > 0", ztest(
    steps_by_each_task['MPS'][2], steps_by_each_task['MPS'][0], alternative='larger'))
print("Fine motor toy, with caregiver, state 3 > 0", ztest(
    steps_by_each_task['MPS'][3], steps_by_each_task['MPS'][0], alternative='larger'))
print("Fine motor toy, with caregiver, state 4 > 0", ztest(
    steps_by_each_task['MPS'][4], steps_by_each_task['MPS'][0], alternative='larger'))
# print("Fine motor toy, state 3 > 4",ztest(steps_by_each_task['NMS'][3] + steps_by_each_task['MPS'][3] + steps_by_each_task['NMS'][4] + steps_by_each_task['MPS'][4],\
#                                            steps_by_each_task['NMS'][1] + steps_by_each_task['MPS'][1] + steps_by_each_task['NMS'][2] + steps_by_each_task['MPS'][2],\
#                                            alternative = 'larger'))

print("\n")
print("Gross motor toy, no caregiver, state 3 > 4", ztest(
    steps_by_each_task['NMM'][3], steps_by_each_task['NMM'][4], alternative='larger'))
print("Gross motor toy, no caregiver, state 3 > 2", ztest(
    steps_by_each_task['NMM'][3], steps_by_each_task['NMM'][2], alternative='larger'))
print("Gross motor toy, no caregiver, state 3 > 1", ztest(
    steps_by_each_task['NMM'][3], steps_by_each_task['NMM'][1], alternative='larger'))
print("Gross motor toy, no caregiver, state 1 > 0", ztest(
    steps_by_each_task['NMM'][1], steps_by_each_task['NMM'][0], alternative='larger'))
print("Gross motor toy, no caregiver, state 2 > 0", ztest(
    steps_by_each_task['NMM'][2], steps_by_each_task['NMM'][0], alternative='larger'))
print("Gross motor toy, no caregiver, state 3 > 0", ztest(
    steps_by_each_task['NMM'][3], steps_by_each_task['NMM'][0], alternative='larger'))
print("Gross motor toy, no caregiver, state 4 > 0", ztest(
    steps_by_each_task['NMM'][4], steps_by_each_task['NMM'][0], alternative='larger'))


print("Gross motor toy, with caregiver, state 3 > 4", ztest(
    steps_by_each_task['MPM'][3], steps_by_each_task['MPM'][4], alternative='larger'))
print("Gross motor toy, with caregiver, state 3 > 2", ztest(
    steps_by_each_task['MPM'][3], steps_by_each_task['MPM'][2], alternative='larger'))
print("Gross motor toy, with caregiver, state 3 > 1", ztest(
    steps_by_each_task['MPM'][3], steps_by_each_task['MPM'][1], alternative='larger'))
print("Gross motor toy, with caregiver, state 1 > 0", ztest(
    steps_by_each_task['MPM'][1], steps_by_each_task['MPM'][0], alternative='larger'))
print("Gross motor toy, with caregiver, state 2 > 0", ztest(
    steps_by_each_task['MPM'][2], steps_by_each_task['MPM'][0], alternative='larger'))
print("Gross motor toy, with caregiver, state 3 > 0", ztest(
    steps_by_each_task['MPM'][3], steps_by_each_task['MPM'][0], alternative='larger'))
print("Gross motor toy, with caregiver, state 4 > 0", ztest(
    steps_by_each_task['MPM'][4], steps_by_each_task['MPM'][0], alternative='larger'))

# print("Gross motor toy, state 3 > 1",ztest(steps_by_each_task['NMM'][3] + steps_by_each_task['MPM'][3] + steps_by_each_task['NMM'][4] + steps_by_each_task['MPM'][4],\
#                                            steps_by_each_task['NMM'][1] + steps_by_each_task['MPM'][1] + steps_by_each_task['NMM'][2] + steps_by_each_task['MPM'][2],\
#                                            alternative = 'larger'))

print("Gross motor toy, state 3 > 0", ztest(steps_by_each_task['NMM'][3] + steps_by_each_task['MPM'][3] + steps_by_each_task['NMM'][3] + steps_by_each_task['MPM'][3],
                                            steps_by_each_task['NMM'][0] + steps_by_each_task['MPM'][0] +
                                            steps_by_each_task['NMM'][0] +
                                            steps_by_each_task['MPM'][0],
                                            alternative='larger'))

# %% compare across states
# for state in range(n_states):
# print('\n')
# print('gross motor toy')
print('state 1 vs state 0')
print(ztest(movement_time_by_each_task['NMM'][1]+movement_time_by_each_task['NMS'][1]+movement_time_by_each_task['MPM'][1]+movement_time_by_each_task['MPS'][1],
            movement_time_by_each_task['NMM'][0]+movement_time_by_each_task['NMS'][0]+movement_time_by_each_task['MPM'][0]+movement_time_by_each_task['MPS'][0], alternative='larger'))

print('state 2 vs state 1')
print(ztest(movement_time_by_each_task['NMM'][2]+movement_time_by_each_task['NMS'][2]+movement_time_by_each_task['MPM'][2]+movement_time_by_each_task['MPS'][2],
            movement_time_by_each_task['NMM'][1]+movement_time_by_each_task['NMS'][1]+movement_time_by_each_task['MPM'][1]+movement_time_by_each_task['MPS'][1], alternative='larger'))
print('state 3 vs state 2')
print(ztest(movement_time_by_each_task['NMM'][3]+movement_time_by_each_task['NMS'][3]+movement_time_by_each_task['MPM'][3]+movement_time_by_each_task['MPS'][3],
            movement_time_by_each_task['NMM'][2]+movement_time_by_each_task['NMS'][2]+movement_time_by_each_task['MPM'][2]+movement_time_by_each_task['MPS'][2], alternative='larger'))

print('state 4 vs state 3')
print(ztest(movement_time_by_each_task['NMM'][4]+movement_time_by_each_task['NMS'][4]+movement_time_by_each_task['MPM'][4]+movement_time_by_each_task['MPS'][4],
            movement_time_by_each_task['NMM'][3]+movement_time_by_each_task['NMS'][3]+movement_time_by_each_task['MPM'][3]+movement_time_by_each_task['MPS'][3], alternative='larger'))

# print('fine motor toy')
# print(ztest(movement_time_by_each_task['NMS'][state], movement_time_by_each_task['MPS'][state], alternative = 'larger'))
