# %%
from scipy.cluster.hierarchy import optimal_leaf_ordering, linkage, leaves_list
from scipy.spatial.kdtree import distance_matrix
from all_visualization_20210824 import rank_state
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
from variables import tasks, state_color_dict_shades
from scipy.cluster.hierarchy import dendrogram
from typing import List

# import scipy.spatial.distance.jensenshannon as jensenshannon
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import matplotlib
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


def DTW(a, b):
    an = len(a)
    bn = len(b)
    cumdist = np.matrix(np.ones((an + 1, bn + 1)) * np.inf)
    cumdist[0, 0] = 0
    # jensenshannon

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min(
                [cumdist[ai, bi + 1], cumdist[ai + 1, bi], cumdist[ai, bi]]
            )
            cost = jensenshannon(a[ai], b[bi])
            # print(ai, bi, cost)
            if np.isnan(cost):
                cost = 0
            cumdist[ai + 1, bi + 1] = cost + minimum_cost
    return cumdist[ai, bi]


def convert_matrix_to_dense(simi_matrix):
    m = simi_matrix.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return simi_matrix[mask]


# %%
with open("../data/interim/20210818_baby_info.pickle", "rb") as f:
    infant_info = pickle.load(f)

# for infant_id, walking_exp in infant_info['walking_exp'].items():
#     ranked_infant_by_walk_exp[infant_id] = walking_exp

ranked_infant_dict = {
    k: v
    for k, v in sorted(infant_info["walking_exp"].items(), key=lambda item: item[1])
}


interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = "n_new_toy_ratio"
with open(
    "../data/result/pickle_files/pred_10s_1.5min_5states_20210907.pickle", "rb"
) as f:
    merged_pred_dict_all = pickle.load(f)
with open(
    "../data/result/pickle_files/pred_prob_10s_1.5min_5states_20210907.pickle", "rb"
) as f:
    all_prob_dict_all = pickle.load(f)

subj_list = np.array(list(all_prob_dict_all["MPS"].keys()))
rank_index = []
for i in ranked_infant_dict.keys():
    idx = np.where(i == subj_list)[0].item()
    rank_index.append(idx)
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


with open("../data/interim/20210718_babymovement.pickle", "rb") as f:
    baby_movement = pickle.load(f)
with open("../data/interim/20210818_baby_info.pickle", "rb") as f:
    infant_info = pickle.load(f)

# %%
# calculating the pairwise similarity between 2 sessions
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
#%%
infant_list = list(all_prob_dict_all[task].keys())
conditions = {0: "MPS", 1: "MPM", 2: "NMS", 3: "NMM"}


def label_function_all(
    id: int,
) -> str:
    if id < 160:
        infant_id = infant_list[id % 40]
        condition_name = conditions[id // 40]
        return condition_name


def label_function_all(
    id: int,
) -> str:
    if id < 160:
        infant_id = infant_list[id % 40]
        condition_name = conditions[id // 40]
        return condition_name
    # else:
    # return '[%d %d %1.2f]' % (id, count, R[n-id,3])

#%%
#%%
big_list_of_state_seq = []
step_per_session = []
step_each_session_per_task = {}
infant_exp = []
for task in tasks:
    step_per_session_this_task = []
    for infant_id, pred_seq in merged_pred_dict_all[task].items():
        # state_list = [state_name_dict[i] for i in pred_seq]
        big_list_of_state_seq.append(pred_seq)
        movement_df = baby_movement[task][infant_id]
        step_per_session.append(movement_df["babymovementSteps"].sum())
        step_per_session_this_task.append(movement_df["babymovementSteps"].sum())
        infant_exp.append(infant_info["walking_exp"][infant_id])
    step_each_session_per_task[task] = step_per_session_this_task

max_ = np.amax(np.array(step_per_session))
# %%
# dense distance matrix
dense_matrix = convert_matrix_to_dense(all_distance)
fig = plt.figure(figsize=(25, 25))
Z = linkage(dense_matrix, "ward", optimal_ordering=True)  # optimal leaf node ordering
result_dict = dendrogram(Z, leaf_label_func=label_function_all, leaf_rotation=90)
optimal_list = leaves_list(optimal_leaf_ordering(Z, dense_matrix))
# print(optimal_list)



#%%
convert_idx_to_condition = {idx: task for idx, task in enumerate(tasks)}

big_task_list = []
infant_id_list = []
# hatch_list = []
for id in result_dict["leaves"]:
    idx = id // 40
    big_task_list.append(convert_idx_to_condition[idx])
    infant_id = infant_list[id % 40]
    condition_name = conditions[id // 40]
    infant_id_list.append(condition_name + str(infant_id))
    # hatch = "/" if idx % 2 == 1 else "None"
#%%
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(label_colors[lbl.get_text()])

#%%
# plot all condition 
fig, axs = plt.subplots(nrows=160, ncols=1, figsize=(160, 80))
for ax_id, pred_seq_idx in enumerate(result_dict["leaves"]):
    state_list = big_list_of_state_seq[pred_seq_idx]
    session_len = len(state_list) if len(state_list) <= 16 else 16
    for i in range(session_len):
        axs[ax_id].add_patch(
            Rectangle(
                (i, 0),
                1,
                5,
                ec="black",
                fc=state_color_dict_shades[str(state_list[i])],
                fill=True,
                alpha=0.7,
            )
        )

    axs[ax_id].set_xticks(np.arange(0, 18, 2))
    axs[ax_id].set_xticklabels("")
    axs[ax_id].set_yticklabels("")
    axs[ax_id].set_xlim(right=16)


    axs[ax_id].yaxis.set_label_position("right")
    axs[-1].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize=102)
    axs[-1].set_xlabel("Minutes", fontsize=102)

plt.show()
plt.close()

#%%

condition_colors = {"MPS": "red", "NMS": "orange", "MPM": "blue", "NMM": "green"}
fig, axs = plt.subplots(nrows=160, ncols=2, figsize=(160, 80))

for ax_id, pred_seq_idx in enumerate(result_dict["leaves"]):
    state_list = big_list_of_state_seq[pred_seq_idx]
    session_len = len(state_list) if len(state_list) <= 16 else 16
    color_list = [
        state_color_dict_shades[str(state_list[i])] for i in range(len(state_list))
    ]
    for i in range(session_len):
        # print(state_color_dict_shades[state_list[i]])
        # print(i)
        axs[ax_id][0].add_patch(
            Rectangle(
                (i, 0),
                1,
                5,
                ec="black",
                fc=state_color_dict_shades[str(state_list[i])],
                fill=True,
                alpha=0.7,
            )
        )
        # if big_task_list[ax_id] == "MPS" or big_task_list[ax_id] == "NMS":
    axs[ax_id][1].barh(
        0, step_per_session[ax_id], color=condition_colors[big_task_list[ax_id]]
    )
    # else:
    #     axs[ax_id][1].barh(
    #         0,
    #         step_per_session[ax_id],
    #         color=condition_colors[big_task_list[ax_id]],
    #         # hatch="/",
    #     )

    axs[ax_id][0].set_xticks(np.arange(0, 18, 2))
    axs[ax_id][0].set_xticklabels("")

    axs[ax_id][0].set_yticklabels("")
    axs[ax_id][0].set_xlim(right=16)
    # axs[ax_id][0].set_ylabel(
    #     infant_id_list[ax_id],
    #     fontsize=50,
    #     rotation=0,
    #     color=condition_colors[big_task_list[ax_id]],
    #     labelpad=80,
    # )

    axs[ax_id][0].yaxis.set_label_position("right")
    # axs[ax_id][0].yaxis.set_label_position("right")
    axs[ax_id][1].set_xlim(right=max_)
    axs[ax_id][1].set_ylabel("")
    axs[ax_id][1].set_xlabel("")

    axs[ax_id][1].set_yticklabels("")
    axs[ax_id][1].set_xticklabels("")

    axs[-1][0].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize=102)
    axs[-1][0].set_xlabel("Minutes", fontsize=102)
    axs[-1][1].set_xlabel("# steps in session", fontsize=102)

    params = {"axes.labelsize": 24, "xtick.labelsize": 102}
    matplotlib.rcParams.update(params)
    plt.subplots_adjust(wspace=0.2)
plt.show()
plt.close()


#%%
state_color_dict_shades = {
    "0": "grey",
    "1": "maroon",
    "2": "salmon",
    "3": "royalblue",
    "4": "midnightblue",
    "5": "midnightblue",
    "6": "midnightblue",
    "7": "blue",
}
fig, axs = plt.subplots(nrows=160, ncols=2, figsize=(160, 80))
max_walking_exp = np.amax(np.array(infant_exp))

for ax_id, pred_seq_idx in enumerate(result_dict["leaves"]):
    state_list = big_list_of_state_seq[pred_seq_idx]
    session_len = len(state_list) if len(state_list) <= 16 else 16
    color_list = [
        state_color_dict_shades[str(state_list[i])] for i in range(len(state_list))
    ]
    for i in range(session_len):
        # print(state_color_dict_shades[state_list[i]])
        # print(i)
        axs[ax_id][0].add_patch(
            Rectangle(
                (i, 0),
                1,
                5,
                ec="black",
                fc=state_color_dict_shades[str(state_list[i])],
                fill=True,
                alpha=0.7,
            )
        )
    axs[ax_id][0].set_xticks(np.arange(0, 18, 2))
    axs[ax_id][0].set_xticklabels("")

    axs[ax_id][0].set_yticklabels("")
    axs[ax_id][0].set_xlim(right=16)

    # axs[ax_id][0].yaxis.set_label_position("right")
    bar_color = "blue" if infant_exp[ax_id] > 90 else "red"
    axs[ax_id][1].barh(0, infant_exp[ax_id], color=bar_color)
    axs[ax_id][1].set_xlim(right=max_walking_exp)
    axs[ax_id][1].set_ylabel("")
    axs[ax_id][1].set_xlabel("")

    axs[ax_id][1].set_yticklabels("")
    axs[ax_id][1].set_xticklabels("")

    axs[-1][0].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize=102)
    axs[-1][0].set_xlabel("Minutes", fontsize=102)
    axs[-1][1].set_xlabel("Walking experience", fontsize=102)

    params = {"axes.labelsize": 24, "xtick.labelsize": 102}
    matplotlib.rcParams.update(params)
plt.show()
plt.close()
#%%

ordered_by_distance = all_distance[np.ix_(result_dict["leaves"], result_dict["leaves"])]
params = {"axes.labelsize": 24, "xtick.labelsize": 12}
matplotlib.rcParams.update(params)
sns.heatmap(ordered_by_distance)
plt.title("Similarity matrix, minimized distances between two sessions")

# %%
all_distance_dict = {}

for idx, task in enumerate(tasks):
    # ax = sns.heatmap(all_distance[idx*40:(idx+1)*40, idx*40:(idx+1)*40])
    all_distance_dict[task] = all_distance[
        idx * 40 : (idx + 1) * 40, idx * 40 : (idx + 1) * 40
    ]

# %%
big_list = []
for i in range(4):
    big_list.extend((np.array(rank_index) + 40 * i).tolist())

ax = sns.heatmap(all_distance[np.ix_(big_list, big_list)])

# %%
# rank infants by walking experience then draw the heat map by each condition
for idx, task in enumerate(tasks):

    curr = all_distance[idx * 40 : (idx + 1) * 40, idx * 40 : (idx + 1) * 40].copy()

    # rearrange by infant walking experience
    curr = curr[np.ix_(rank_index, rank_index)]
    ax = sns.heatmap(curr)
    fig_name = (
        "./figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states/similarity_matrix_"
        + task
        + ".png"
    )
    plt.show()
    plt.savefig(fig_name)
    plt.close()


# %%
with open("../data/interim/20210718_babymovement.pickle", "rb") as f:
    baby_movement = pickle.load(f)
total_num_steps_dict = {}
rank_infant_by_step_take_each_task = {}
all_step_indices = []
for task_idx, (task, movement_dict) in enumerate(baby_movement.items()):
    each_task_infant_order = []
    each_task_dict = {}
    for subj, movement_df in movement_dict.items():
        each_task_dict[subj] = movement_df["babymovementSteps"].sum()
    ranked_step_dict = {
        k: v for k, v in sorted(each_task_dict.items(), key=lambda item: item[1])
    }
    total_num_steps_dict[task] = ranked_step_dict

    for i in ranked_infant_dict.keys():
        idx = np.where(i == subj_list)[0].item()
        each_task_infant_order.append(idx)
    rank_infant_by_step_take_each_task[task] = each_task_infant_order
    all_step_indices.extend((np.array(each_task_infant_order) + task_idx * 40).tolist())


# %%
# all_step_indices
all_distance_by_step = all_distance.copy()
all_distance_by_step = all_distance_by_step[np.ix_(all_step_indices, all_step_indices)]
plt.figure(figsize=(10, 10))
ax = sns.heatmap(all_distance_by_step)

# %%
for idx, task in enumerate(tasks):
    curr = all_distance[idx * 40 : (idx + 1) * 40, idx * 40 : (idx + 1) * 40].copy()
    # rearrange by infant experience
    index_by_total_step = rank_infant_by_step_take_each_task[task]
    curr = curr[np.ix_(index_by_total_step, index_by_total_step)]
    ax = sns.heatmap(curr)
    fig_name = (
        "./figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states/similarity_matrix_"
        + task
        + "_by_n_steps.png"
    )
    plt.show()
    plt.savefig(fig_name)
    plt.close()



#%%
# Draw each task with walking experience 
for task_idx, task in enumerate(tasks):
    curr = all_distance[
        task_idx * 40 : (task_idx + 1) * 40, task_idx * 40 : (task_idx + 1) * 40
    ].copy()
    dense_matrix_current_condition = convert_matrix_to_dense(curr)
    fig = plt.figure(figsize=(15, 15))
    Z = linkage(dense_matrix_current_condition, "ward", optimal_ordering=True)
    result_dict = dendrogram(Z)
    rearranged_idx = result_dict["leaves"]
    ordered_by_distance_task = curr[np.ix_(rearranged_idx, rearranged_idx)]
    # sns.heatmap(ordered_by_distance_task)
    plt.close()

    # plot against walking_exp
    fig, axs = plt.subplots(nrows=40, ncols=2, figsize=(160, 80))
    max_walking_exp = np.amax(np.array(infant_exp))
    pred_this_task = list(merged_pred_dict_all[task].values())
    walking_exp = np.array(list(infant_info["walking_exp"].values()))
    arranged_walking_exp = walking_exp[rearranged_idx]
    for id_of_idx, ax_id in enumerate(rearranged_idx):
        state_list = pred_this_task[ax_id]
        session_len = len(state_list) if len(state_list) <= 16 else 16
        color_list = [
            state_color_dict_shades[str(state_list[i])] for i in range(len(state_list))
        ]
        bar_color = "blue" if arranged_walking_exp[id_of_idx] > 90 else "red"

        for i in range(session_len):
            # print(state_color_dict_shades[state_list[i]])
            # print(i)
            axs[id_of_idx][0].add_patch(
                Rectangle(
                    (i, 0),
                    1,
                    5,
                    ec="black",
                    fc=state_color_dict_shades[str(state_list[i])],
                    fill=True,
                    alpha=0.7,
                )
            )

        axs[id_of_idx][0].set_xticks(np.arange(0, 18, 2))
        axs[id_of_idx][0].set_xticklabels("")

        axs[id_of_idx][0].set_yticklabels("")
        axs[id_of_idx][0].set_xlim(right=16)
        axs[-1][0].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize=102)
        axs[-1][0].set_xlabel("Minutes", fontsize=102)
        axs[id_of_idx][1].barh(0, width=arranged_walking_exp[id_of_idx], color=bar_color)

        axs[id_of_idx][1].set_ylabel("")
        axs[id_of_idx][1].set_xlabel("")

        axs[id_of_idx][1].set_yticklabels("")
        axs[id_of_idx][1].set_xticklabels("")
        axs[id_of_idx][1].set_xlim(right=max_walking_exp)
        axs[-1][1].set_xlabel("Walking experience", fontsize=102)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

#%%
# Draw each task with n_steps
for task_idx, task in enumerate(tasks):
    curr = all_distance[
        task_idx * 40 : (task_idx + 1) * 40, task_idx * 40 : (task_idx + 1) * 40
    ].copy()
    dense_matrix_current_condition = convert_matrix_to_dense(curr)
    fig = plt.figure(figsize=(15, 15))
    Z = linkage(dense_matrix_current_condition, "ward", optimal_ordering=True)
    result_dict = dendrogram(Z)
    rearranged_idx = result_dict["leaves"]
    ordered_by_distance_task = curr[np.ix_(rearranged_idx, rearranged_idx)]
    # sns.heatmap(ordered_by_distance_task)
    plt.close()

    # plot against walking_exp
    fig, axs = plt.subplots(nrows=40, ncols=2, figsize=(160, 80))
    max_n_steps = np.amax(np.array(step_per_session))
    pred_this_task = list(merged_pred_dict_all[task].values())
    for id_of_idx, ax_id in enumerate(rearranged_idx):
        state_list = pred_this_task[ax_id]
        session_len = len(state_list) if len(state_list) <= 16 else 16
        color_list = [
            state_color_dict_shades[str(state_list[i])] for i in range(len(state_list))
        ]
        # bar_color = "blue" if arranged_walking_exp[id_of_idx] > 90 else "red"

        for i in range(session_len):
            # print(state_color_dict_shades[state_list[i]])
            # print(i)
            axs[id_of_idx][0].add_patch(
                Rectangle(
                    (i, 0),
                    1,
                    5,
                    ec="black",
                    fc=state_color_dict_shades[str(state_list[i])],
                    fill=True,
                    alpha=0.7,
                )
            )

        axs[id_of_idx][0].set_xticks(np.arange(0, 18, 2))
        axs[id_of_idx][0].set_xticklabels("")

        axs[id_of_idx][0].set_yticklabels("")
        axs[id_of_idx][0].set_xlim(right=16)
        axs[-1][0].set_xticklabels([str(x) for x in np.arange(0, 9, 1)], fontsize=102)
        axs[-1][0].set_xlabel("Minutes", fontsize=102)
        axs[id_of_idx][1].barh(0, width=step_each_session_per_task[task][ax_id])

        axs[id_of_idx][1].set_ylabel("")
        axs[id_of_idx][1].set_xlabel("")

        axs[id_of_idx][1].set_yticklabels("")
        axs[id_of_idx][1].set_xticklabels("")
        axs[id_of_idx][1].set_xlim(right=max_n_steps)
        axs[-1][1].set_xlabel("Number of steps in session", fontsize=102)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()