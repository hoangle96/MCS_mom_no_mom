#%%
import pickle
from pathlib import Path 
import matplotlib.pyplot as plt 
import numpy as np
from variables import tasks, toy_to_task_dict
import pandas as pd
from feature_engineering import get_first_last_time, rank_toy_local
#%%
no_ops_threshold = 10
interval_length = .5
with open("../data/interim/20210824_"+str(no_ops_threshold)+"_no_ops_threshold_feature_dict_with_n_new_toy_ratio_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict_with_n_new_toy_ratio = pickle.load(f)
with open('../data/interim/20210824_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    task_to_storing_dict = pickle.load(f)
with open('../data/interim/20210824_floor_time.pickle', 'rb') as f:
    floor_time = pickle.load(f)

# %%
feature_vector = np.empty((0, 4))
for task in tasks:
    for subj in feature_dict_with_n_new_toy_ratio[task].keys():
        feature_vector = np.vstack((feature_vector, feature_dict_with_n_new_toy_ratio[task][subj][0]))

# %%
plt.hist(feature_vector[:,0])
# %%
plt.hist(feature_vector[:,1])

# %%
plt.hist(feature_vector[:,2])

# %%
plt.hist(feature_vector[:,3])

# %%
no_ops_threshold = 10
interval_length = 1.5
with open("../data/interim/20210824_"+str(no_ops_threshold)+"_no_ops_threshold_feature_dict_with_n_new_toy_ratio_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict_with_n_new_toy_ratio_og = pickle.load(f)
# %%
feature_vector_og = np.empty((0, 4))
for task in tasks:
    for subj in feature_dict_with_n_new_toy_ratio_og[task].keys():
        feature_vector_og = np.vstack((feature_vector_og, feature_dict_with_n_new_toy_ratio_og[task][subj][0]))
# %%
fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey=True)
plt.suptitle("Histogram of number of toy switches")
plt.xlabel("# toy switches")
cnt, val = np.histogram(feature_vector_og[:,0], bins = [0, 10, 20, 30, 40])
axs[0].bar(val[1:], cnt/cnt.sum()*100, width = 5)
axs[0].grid(False)
axs[0].set_title("Interval length of 1.5 min")
cnt, val = np.histogram(feature_vector[:,0], bins = [0, 10, 20, 30, 40])
axs[1].bar(val[1:], cnt/cnt.sum()*100, width = 5)
axs[1].set_title("Interval length of 30s")
axs[1].grid(False)
axs[0].set_ylabel("%")

# %%
fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey=True, figsize = (10, 10))
plt.suptitle("Histogram of number of toys")
plt.xlabel("# toys")
val, cnt = np.unique(feature_vector_og[:,1], return_counts= True)
print(val)
axs[0].bar(val, cnt/cnt.sum()*100)
axs[0].grid(False)
axs[0].set_title("Interval length of 1.5 min")
axs[0].set_xticks(val)
axs[0].set_ylabel("%")
axs[0].set_xticklabels([str(int(x)) for x in val])

val, cnt = np.unique(feature_vector[:,1], return_counts= True)
axs[1].bar(val, cnt/cnt.sum()*100)
axs[1].set_title("Interval length of 30s")
axs[1].grid(False)
axs[1].set_xticks(val)
axs[1].set_xticklabels([str(int(x)) for x in val])


# %%
fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey=True)
plt.suptitle("Histogram of number of new toys ratio")
plt.xlabel("new toys ratio")
cnt, val = np.histogram(feature_vector_og[:,2], bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
axs[0].bar(val[1:], cnt/cnt.sum()*100)
axs[0].grid(False)
axs[0].set_title("Interval length of 1.5 min")
cnt, val = np.histogram(feature_vector[:,2], bins = [0, 0.2, 0.4, 0.6, 0.8, 1])
print(val)
axs[1].bar(val[1:], cnt/cnt.sum()*100)
axs[1].set_title("Interval length of 30s")
axs[1].grid(False)
axs[0].set_ylabel("%")
# axs[1].set_xticks(val)
# axs[1].set_xticklabels([str(int(x)) for x in val])
# 
# # %%
plt.hist(feature_vector_og[:,1])
# %%
plt.hist(feature_vector_og[:,2])

# %%

plt.hist(feature_vector_og[:,3])

# %%
all_toy_play_time_vector = np.empty((0, 7))
for task_idx, task in enumerate(['MPM', "NMM"]):
    for subj, df in task_to_storing_dict[task].items():
        sess_len = 0
        for f_time in floor_time[subj][task]:
            sess_len += f_time[-1]-f_time[0]

        subj_df = pd.DataFrame()
        for df_ in task_to_storing_dict[task][subj]:
            subj_df = pd.concat([subj_df, df_])
        start_time, end_time = get_first_last_time(subj_df)
        toy_rank = rank_toy_local(subj_df, toy_to_task_dict[task], start_time, end_time)
        toy_play_time = np.array(list(toy_rank.values())).reshape((-1, 7))/sess_len
        all_toy_play_time_vector = np.vstack((all_toy_play_time_vector, toy_play_time))
plt.hist(all_toy_play_time_vector[:,0])
plt.show()
plt.close()
plt.hist(all_toy_play_time_vector[:,-1])

# %%
toy_rank
# %%
sess_len
# %%
toy_play_time
# %%
print(len(all_toy_play_time_vector[:,0]))
# %%
print(all_toy_play_time_vector[:,1])

# %%
