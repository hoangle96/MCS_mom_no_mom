# %%
from scipy.cluster.hierarchy import optimal_leaf_ordering, linkage
from all_visualization_20210824 import rank_state
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
from variables import tasks

# import scipy.spatial.distance.jensenshannon as jensenshannon
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw

# def JSD(P, Q):
#     _P = P / norm(P, ord=1)
#     _Q = Q / norm(Q, ord=1)
#     _M = 0.5 * (_P + _Q)
#     return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def DTW(a, b):
    an = len(a)
    bn = len(b)
    cumdist = np.matrix(np.ones((an+1, bn+1)) * np.inf)
    cumdist[0, 0] = 0
    # jensenshannon

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cost = jensenshannon(a[ai], b[bi])
            # print(ai, bi, cost)
            if np.isnan(cost):
                cost = 0
            cumdist[ai+1, bi+1] = cost + minimum_cost
    return cumdist[ai, bi]


def convert_matrix_to_dense(simi_matrix):
    m = simi_matrix.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return simi_matrix[mask]


# %%
with open('../data/interim/20210818_baby_info.pickle', 'rb') as f:
    infant_info = pickle.load(f)

# for infant_id, walking_exp in infant_info['walking_exp'].items():
#     ranked_infant_by_walk_exp[infant_id] = walking_exp

ranked_infant_dict = {k: v for k, v in sorted(
    infant_info['walking_exp'].items(), key=lambda item: item[1])}
ranked_infant_dict

# ranked_infant_id = {k: v for k, v in sorted(ranked_infant_dict.items(), key=lambda item: item[0])}
# print(ranked_infant_id)


# %%
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
with open('../data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_all_pred_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    all_prob_dict_all = pickle.load(f)
with open('../data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_pred_dict_all = pickle.load(f)
# print(len(all_prob_dict_all["MPS"][1]))
subj_list = np.array(list(all_prob_dict_all['MPS'].keys()))
rank_index = []
for i in ranked_infant_dict.keys():
    idx = np.where(i == subj_list)[0].item()
    rank_index.append(idx)
# print(DTW(all_prob_dict_all["MPS"][1], all_prob_dict_all["MPS"][1]))
# print(DTW(all_prob_dict_all["MPS"][1], all_prob_dict_all["MPM"][3]))
#%%
all_prob_dict_all["MPS"][1]
# merged_pred_dict_all["MPS"][1]
# %%
all_distance = np.empty((160, 160))

row_idx = 0
for task in tasks:
    for subj, seq in all_prob_dict_all[task].items():
        col_idx = 0
        for task_ in tasks:
            for subj_, seq_ in all_prob_dict_all[task_].items():
                distance = DTW(seq, seq_)
                all_distance[row_idx][col_idx] = distance
                col_idx += 1
        row_idx += 1


# %%
# dense distance matrix
dense_matrix = convert_matrix_to_dense(all_distance)
Z = linkage(dense_matrix, 'ward', optimal_ordering=True)
print(Z)
# %%
m = 4
for i in range(m):
    for j in range(i+1, m):
        pos = (4*i+j)-((i+2)*(i+1))/2
        print(i, j, pos)
# %%
print(len(list(all_prob_dict_all['MPS'].keys())))
# %%
all_distance_dict = {}

for idx, task in enumerate(tasks):
    # ax = sns.heatmap(all_distance[idx*40:(idx+1)*40, idx*40:(idx+1)*40])
    all_distance_dict[task] = all_distance[idx *
                                           40:(idx+1)*40, idx*40:(idx+1)*40]
# %%
a = np.array(all_prob_dict_all["MPS"][1]).reshape((-1, 5))
b = np.array(all_prob_dict_all["MPS"][2]).reshape((-1, 5))

jensenshannon(a, b)

# %%
big_list = []
for i in range(4):
    big_list.extend((np.array(rank_index)+40*i).tolist())

ax = sns.heatmap(all_distance[np.ix_(big_list, big_list)])

# %%
# rank infants by walking experience then draw the heat map
for idx, task in enumerate(tasks):
    curr = all_distance[idx*40:(idx+1)*40, idx*40:(idx+1)*40].copy()
    # rearrange by infant walking experience
    curr = curr[np.ix_(rank_index, rank_index)]
    ax = sns.heatmap(curr)
    fig_name = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_' + \
        str(no_ops_time)+'/window_size_'+str(interval_length)+'/' + \
        str(n_states)+"_states/similarity_matrix_"+task+".png"
    plt.show()
    plt.savefig(fig_name)
    plt.close()

# %%
ax = sns.heatmap(curr)
np.array(())

# %%
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
    ranked_step_dict = {k: v for k, v in sorted(
        each_task_dict.items(), key=lambda item: item[1])}
    total_num_steps_dict[task] = ranked_step_dict

    for i in ranked_infant_dict.keys():
        idx = np.where(i == subj_list)[0].item()
        each_task_infant_order.append(idx)
    rank_infant_by_step_take_each_task[task] = each_task_infant_order
    all_step_indices.extend(
        (np.array(each_task_infant_order)+task_idx*40).tolist())


# %%
# all_step_indices
all_distance_by_step = all_distance.copy()
all_distance_by_step = all_distance_by_step[np.ix_(
    all_step_indices, all_step_indices)]
plt.figure(figsize=(10, 10))
ax = sns.heatmap(all_distance_by_step)

# %%

for idx, task in enumerate(tasks):
    curr = all_distance[idx*40:(idx+1)*40, idx*40:(idx+1)*40].copy()
    # rearrange by infant experience
    index_by_total_step = rank_infant_by_step_take_each_task[task]
    curr = curr[np.ix_(index_by_total_step, index_by_total_step)]
    ax = sns.heatmap(curr)
    fig_name = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(
        interval_length)+'/'+str(n_states)+"_states/similarity_matrix_"+task+"_by_n_steps.png"
    plt.show()
    plt.savefig(fig_name)
    plt.close()

# %%
# cal bigram
model_file_name = "model_20210907_"+feature_set+"_" + \
    str(interval_length)+"_interval_length_"+str(no_ops_time) + \
    "_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('../models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)
trans_mat = model.dense_transition_matrix()[:n_states, :n_states]
init_prob = model.dense_transition_matrix()[n_states+1, :n_states]

# convert to bigram using the transition matrix
all_bigram_seq = []
bigram_prob_dict = {}
for task in tasks:
    each_task = {}
    for subj, seq in merged_pred_dict_all[task].items():
        new_seq = []
        for idx, state in enumerate(seq):
            prob = all_prob_dict_all[task][subj][idx]
            if idx == 0:
                new_seq.append(prob)
            else:
                bi_gram = trans_mat[state]
                # bi_gram = bi_gram/bi_gram.sum()
                new_seq.append(bi_gram)
        each_task[subj] = new_seq
        all_bigram_seq.append(new_seq)
    bigram_prob_dict[task] = each_task

bigram_distance = np.empty((160, 160))
for row_idx, seq in enumerate(all_bigram_seq):
    for col_idx, seq_ in enumerate(all_bigram_seq):
        distance = DTW(seq, seq_)
        # if np.isnan(distance):
        # distance = 0
        bigram_distance[row_idx, col_idx] = distance


# %%
# ax = sns.heatmap(bigram_distance)
for idx, task in enumerate(tasks):
    curr = bigram_distance[idx*40:(idx+1)*40, idx*40:(idx+1)*40].copy()
    # rearrange by infant experience
    index_by_total_step = rank_infant_by_step_take_each_task[task]
    curr = curr[np.ix_(index_by_total_step, index_by_total_step)]
    ax = sns.heatmap(curr)
    fig_name = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(
        interval_length)+'/'+str(n_states)+"_states/similarity_matrix_bi_gram_"+task+"_by_n_steps.png"
    plt.show()
    plt.savefig(fig_name)
    plt.close()
# %%
for idx, task in enumerate(tasks):
    curr = bigram_distance[idx*40:(idx+1)*40, idx*40:(idx+1)*40].copy()
    # rearrange by infant experience
    # index_by_total_step = rank_infant_by_step_take_each_task[task]
    curr = curr[np.ix_(rank_index, rank_index)]
    ax = sns.heatmap(curr)
    fig_name = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(
        interval_length)+'/'+str(n_states)+"_states/similarity_matrix_bi_gram_"+task+"_by_walking_exp.png"
    plt.show()
    plt.savefig(fig_name)
    plt.close()


# %%
a = np.array([0, 0, 0, 0, 0.1])
b = np.array([0, 0, 0.1, 0, 0.5])

print(jensenshannon(a, b))

for i in all_prob_dict_all["NMS"].keys():
    a = all_prob_dict_all["NMS"][i]
    for j in all_prob_dict_all["NMS"].keys():
        b = all_prob_dict_all["NMS"][j]
        print(i, j, DTW(a, b))

# a = all_prob_dict_all["NMS"][list(all_prob_dict_all["NMS"].keys())[10]]
# b = all_prob_dict_all["NMS"][1]
# DTW(a, b)

# %%
# for i in all_prob_dict_all["NMS"].keys():
a = all_prob_dict_all["NMS"][12]
b = all_prob_dict_all["NMS"][40]
# print(a)
# print(b)
DTW(a, b)
# print(len(a))
# %%
print(a[5])
print(b[7])

print(jensenshannon(a[5], b[7]))


# %%
