#%%
import numpy as np
import pandas as pd
from variables import (
    tasks,
    condition_name,
    state_color_dict_shades,
    stationary_toys_list,
    mobile_toys_list,
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
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec


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
                state_label_n_toy_dict[idx] = np.dot(
                    np.array(
                        list(s.distribution.parameters[0][1].parameters[0].values())
                    ),
                    np.array(
                        list(s.distribution.parameters[0][1].parameters[0].keys())
                    ).T,
                ) + np.dot(
                    np.array(
                        list(s.distribution.parameters[0][2].parameters[0].values())
                    ),
                    np.array(
                        list(s.distribution.parameters[0][2].parameters[0].keys())
                    ).T,
                )

    # print(state_label_n_toy_dict)
    ranked_dict = {
        k: v
        for k, v in sorted(state_label_n_toy_dict.items(), key=lambda item: item[1])
    }
    return {v: str(k) for k, v in enumerate(ranked_dict.keys())}


# %%
shift = 0.5
mobile_toys_list.append("no_toy")
stationary_toys_list.append("no_toy")
feature_set = "n_new_toy_ratio"
no_ops_time = 10
interval_length = 1.5
n_states = 5

with open(
    "../data/interim/20210907_"
    + feature_set
    + "_"
    + str(no_ops_time)
    + "_no_ops_threshold_discretized_input_list_"
    + str(interval_length)
    + "_min.pickle",
    "rb",
) as f:
    discretized_input_list = pickle.load(f)

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


x_ticks_dict = {
    0: ["[0, 4)", "[4, 8)", "[8, 12)", "[12+"],
    1: ["0", "1", "2", "3", "4+"],
    2: [".2", ".4", ".6", ".8", "1"],
    3: [".2", ".4", ".6", ".8", "1"],
}
feature_names = ["# toys switches", "# toys", "# new toys ratio", "fav toy ratio"]
feature_values = {0: range(1, 5), 1: range(5), 2: range(1, 6), 3: range(1, 6)}

n_features = 4
flatten_pred = []
flatten_pred_dict = {}
flatten_pred_dict_cg = {"without_cg": [], "with_cg": []}
flatten_pred_dict_toy_set = {"mobile": [], "fine": []}

Path(
    "../figures/hmm/20210907/"
    + feature_set
    + "/no_ops_threshold_"
    + str(no_ops_time)
    + "/window_size_"
    + str(interval_length)
    + "/"
    + str(n_states)
    + "_states/"
).mkdir(parents=True, exist_ok=True)
for task in tasks:
    flatten_pred_dict[task] = []
    if task in ["MPM", "NMM"]:
        toy_task = "mobile"
    else:
        toy_task = "fine"

    if task in ["MPS", "MPM"]:
        mom_task = "with_cg"
    else:
        mom_task = "without_cg"

    task_specific_pred_dict = pred_dict[task]
    for subj, subj_dict in task_specific_pred_dict.items():
        for shift_time, pred in subj_dict.items():
            new_pred = [state_name_dict[x] for x in pred]
            pred = np.array(new_pred).astype(int)
            flatten_pred.extend(pred)
            flatten_pred_dict[task].extend(pred)
            flatten_pred_dict_cg[mom_task].extend(pred)
            flatten_pred_dict_toy_set[toy_task].extend(pred)


# fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_mom.png"
# draw_state_distribution(flatten_pred_dict_cg['with_cg'], n_states, state_name_dict, "With caregiver, both toy sets", state_color_dict_shades, fig_path)

# fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_no_mom.png"
# draw_state_distribution(flatten_pred_dict_cg['without_cg'], n_states, state_name_dict, "Without caregiver, both toy sets", state_color_dict_shades, fig_path)

# fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_mobile_toy.png"
# draw_state_distribution(flatten_pred_dict_toy_set['mobile'], n_states, state_name_dict, "Gross-motor toy sets", state_color_dict_shades, fig_path)

# fig_path = '../figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_stationary_toys.png"
# draw_state_distribution(flatten_pred_dict_toy_set['fine'], n_states, state_name_dict, "Fine-motor toy sets", state_color_dict_shades, fig_path)

#%%
discretized_input_list
#%%
flatten_pred_dict_pct = {}

for task in tasks:
    val, cnt = np.unique(
        np.array(flatten_pred_dict[task]).astype(int), return_counts=True
    )
    pct = cnt / cnt.sum()
    task_state_pct = {v: pct[idx] for idx, v in enumerate(val)}
    for i in range(n_states):
        if i not in task_state_pct.keys():
            task_state_pct[i] = 0
    flatten_pred_dict_pct[task] = task_state_pct

#%%
flatten_pred_dict_pct
# %%


def draw_state_distribution(flatten_pred_dict):
    plt.style.use("seaborn")
    task_edge_color = {"MPS": "r", "MPM": "b", "NMS": "r", "NMM": "b"}
    task_face_color = {"MPS": "r", "MPM": "b", "NMS": "none", "NMM": "none"}
    task_linestyle = {"MPS": "-", "MPM": "-", "NMS": "--", "NMM": "--"}
    task_fill = {"MPS": True, "MPM": True, "NMS": False, "NMM": False}
    task_fill_style = {"MPS": "full", "MPM": "full", "NMS": "none", "NMM": "none"}
    task_name = {
        "MPS": "With caregivers, fine motor toys",
        "MPM": "With caregivers, gross motor toys",
        "NMS": "Without caregivers, fine motor toys",
        "NMM": "Without caregivers, gross motor toys",
    }

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), facecolor="white")
    plt.suptitle(
        "Marginal distribution of each state in different play conditions", fontsize=34
    )
    handles = []

    for task in tasks:
        handles.append(
            Line2D(
                [0],
                [0],
                label=task_name[task],
                markerfacecolor=task_edge_color[task],
                markeredgecolor=task_edge_color[task],
                # fillstyle= task_fill_style[task],
                c=task_edge_color[task],
                ls=task_linestyle[task],
                lw=3,
                markersize=40,
            )
        )
    fig.legend(handles=handles, bbox_to_anchor=(1.52, 1), fontsize=28)

    for state in range(1, 5):
        col_idx = 0 if state % 2 != 0 else 1
        row_idx = 1 if state / 2 > 1 else 0

        for task_idx, task in enumerate(tasks):
            axs[row_idx][col_idx].bar(
                x=task_idx,
                height=flatten_pred_dict[task][state],
                edgecolor=task_edge_color[task],
                linestyle=task_linestyle[task],
                facecolor=task_edge_color[task],
                fill=task_fill[task],
                lw=3,
            )

        # x-axis
        axs[row_idx][col_idx].set_xticks(range(4))
        axs[row_idx][col_idx].set_xticklabels(["" for x in range(1, 5)], fontsize=26)
        # axs[row_idx][col_idx].set_xlabel("States", fontsize = 18)

        # y-axis
        axs[row_idx][col_idx].set_yticks(np.arange(0, 0.9, 0.1))
        axs[row_idx][col_idx].set_yticklabels(
            [str(int(x * 100)) for x in np.arange(0, 0.9, 0.1)], fontsize=26
        )
        axs[row_idx][col_idx].set_ylabel("% session", fontsize=24)
        axs[row_idx][col_idx].set_ylim(top=0.5)

        axs[row_idx][col_idx].set_title("State " + str(state), fontsize=32)
        axs[row_idx][col_idx].grid(axis="x")

    # plt.savefig(file_path)
    plt.tight_layout()
    plt.show()
    plt.close()


#%%
stationary_distribution = {
    "MPS": {0: 0.00272, 1: 0.24953, 2: 0.22605, 3: 0.17558, 4: 0.06098},
    "MPM": {0: 0.04076, 1: 0.17667, 2: 0.17980, 3: 0.30600, 4: 0.36166},
    "NMS": {0: 0.12724, 1: 0.17621, 2: 0.39385, 3: 0.20446, 4: 0.08938},
    "NMM": {0: 0.15985, 1: 0.12438, 2: 0.24080, 3: 0.25129, 4: 0.24817},
}

#%%
def draw_marginal_stationary_distribution(
    tasks, marginal_dist, stationary_dist, n_states, state_color_dict
):
    plt.style.use("default")
    task_name = {
        "MPS": "With caregivers, fine motor toys",
        "MPM": "With caregivers, gross motor toys",
        "NMS": "Without caregivers, fine motor toys",
        "NMM": "Without caregivers, gross motor toys",
    }
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharey="row", figsize=(14, 14), facecolor="white"
    )

    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 3
    handles = [
        Patch(facecolor=state_color_dict["3"]),
        Patch(facecolor=state_color_dict["3"], hatch="/"),
    ]
    fig.legend(
        handles=handles,
        labels=["Marginal distribution", "Stationary distribution"],
        bbox_to_anchor=(0.9, 0.95),
        fontsize=22,
    )

    for task_idx, task in enumerate(tasks):
        col_idx = 0 if task_idx % 2 != 0 else 1
        row_idx = 0 if task_idx / 2 < 1 else 1
        for state in range(n_states):
            axs[row_idx][col_idx].bar(
                x=state * 2.5,
                height=stationary_dist[task][state],
                color=state_color_dict[str(state)],
            )
            axs[row_idx][col_idx].bar(
                x=state * 2.5 + 1,
                height=marginal_dist[task][state],
                color=state_color_dict[str(state)],
                hatch="/",
            )
            axs[row_idx][col_idx].set_title(task_name[task], fontsize=28)
        axs[row_idx][col_idx].grid(axis="y", c="black")
        axs[row_idx][col_idx].tick_params(axis="x", which="minor", bottom=True)
        axs[row_idx][col_idx].grid(axis="x", c="black", which="minor", b=True)
        axs[row_idx][col_idx].set_facecolor("white")

        axs[row_idx][col_idx].set_xticks(np.arange(0.5, 11, 2.5))
        axs[row_idx][col_idx].set_xticklabels(
            [str(state) for state in range(n_states)], fontsize=28
        )
        axs[row_idx][col_idx].set_xlabel("States", fontsize=28)

        # y-axis
        axs[row_idx][col_idx].set_yticks(np.arange(0, 0.9, 0.1))
        axs[row_idx][col_idx].set_yticklabels(
            [str(int(x * 100)) for x in np.arange(0, 0.9, 0.1)], fontsize=24
        )
        axs[row_idx][col_idx].set_ylim(top=0.45)
        for x_pos in [1.75, 4.25, 6.75, 9.25]:
            axs[row_idx][col_idx].axvline(x=x_pos, c="black")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.show()


draw_marginal_stationary_distribution(
    ["NMM", "MPM", "MPS", "NMS"],
    flatten_pred_dict_pct,
    stationary_distribution,
    n_states,
    state_color_dict_shades,
)
#%%
flatten_pred
# %%
# def draw_distribution(
#     n_features,
#     state_name_dict,
#     discretized_input_list,
#     flatten_pred,
#     title,
#     feature_names,
#     x_ticks_dict,
#     feature_values,
#     state_color_dict_shades,
#     fig_path,
# ):
plt.style.use("default")

fig = plt.figure(figsize=(40, 15))
# plt.suptitle("Emission distribution for each state", fontsize=40)
# fig5 = plt.figure()
outer = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.4)
plt.rcParams["figure.constrained_layout.use"] = True
for i in range(4):  # i == state
    state = i + 1
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[i], wspace=0.1, hspace=0.02
    )

    for j in range(4):  # j == feature
        feature = discretized_input_list.T[j]
        # print(feature)
        unique, cnt = np.unique(
            feature[np.array(flatten_pred) == state], return_counts=True
        )
        ax = plt.Subplot(fig, inner[j])
        ax.grid(axis="y", c="black")
        # ax.grid(axis="x", b="black")
        if j == 1:
            vert_x = [0.5, 1.5, 2.5, 3.5, 4.5]
        else:
            vert_x = [1.5, 2.5, 3.5, 4.5]
        for x_pos in vert_x:
            ax.axvline(x=x_pos, c="black")
        all_unique = np.unique(feature)

        x_labels = x_ticks_dict[j]
        x_vals = feature_values[j]
        height = cnt / cnt.sum()
        cnt_dict = {k: v for k, v in zip(unique, height)}

        final_val = []
        final_height = []

        for val in x_vals:

            final_val.append(val)

            if val in cnt_dict.keys():
                final_height.append(cnt_dict[val])
            else:
                final_height.append(0)
        ax.bar(final_val, final_height, color=state_color_dict_shades[str(state)])
        ax.set_ylim(top=1)

        ax.set_xticks([x for x in x_vals])
        ax.set_xticklabels(labels=x_labels, fontsize=22)

        if j == 0:
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_yticklabels(
                labels=[str(np.around(y_i, 1)) for y_i in np.arange(0, 1.1, 0.2)],
                fontsize=28,
            )
            ax.tick_params("y", left=True, labelleft=True)
        else:
            ax.set_yticklabels(labels=[""] * 6, fontsize=28)
        if i == 0:
            ax.set_xlabel(feature_names[j], fontsize=32)
            ax.xaxis.set_label_position("top")

        ax.set_facecolor("white")

        fig.add_subplot(ax)
        # print(i%2)
    # print(i%2, 0.25 * (i % 2), 0.01 * (i%2))
    col_offset = 0.5 if i // 2 == 0 else 0.05
    row_offset = 0.3 if i % 2 == 0 else 0.73

    fig.text(row_offset, col_offset, "State " + str(i + 1), ha="center", fontsize=32)

plt.tight_layout()
fig.show()