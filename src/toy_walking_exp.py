# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
import pickle
from all_visualization_20210824 import rank_state
from variables import (
    tasks,
    toys_list,
    stationary_toys_list,
    mobile_toys_list,
    toy_colors_dict,
)

# %%
interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = "n_new_toy_ratio"

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

with open("../data/interim/20210818_baby_info.pickle", "rb") as f:
    infant_info = pickle.load(f)

with open(
    "../data/interim/20210907"
    + feature_set
    + "_"
    + str(no_ops_time)
    + "_no_ops_theshold_"
    + str(n_states)
    + "_states_toy_pred_dict_"
    + str(interval_length)
    + "_min.pickle",
    "rb",
) as f:
    toy_pred_list = pickle.load(f)

subj_list = list(infant_info["walking_exp"].keys())

# %%
each_cond_time_novice_exp = {"novice": {}, "exp": {}}
for task in tasks:
    each_cond_time_novice_exp["novice"][task] = {}
    each_cond_time_novice_exp["exp"][task] = {}
    for state in state_name_dict.values():
        each_cond_time_novice_exp["novice"][task][state] = {}
        each_cond_time_novice_exp["exp"][task][state] = {}
        if task in ["MPS", "NMS"]:
            toys = stationary_toys_list
        elif task in ["MPM", "NMM"]:
            toys = mobile_toys_list

        for toy in toys:
            each_cond_time_novice_exp["novice"][task][state][toy] = []
            each_cond_time_novice_exp["exp"][task][state][toy] = []

for task in tasks:
    if task in ["MPS", "NMS"]:
        toys = stationary_toys_list
    elif task in ["MPM", "NMM"]:
        toys = mobile_toys_list
    for subj in subj_list:
        if infant_info["walking_exp"][subj] >= 90:
            walk_exp_indicator = "exp"
        else:
            walk_exp_indicator = "novice"

        df_ = toy_pred_list[task][subj].copy()
        df_ = df_.explode("toys")
        df_["toys"] = df_["toys"].replace({"no_ops": "no_toy"})
        df_["duration"] = df_["offset"] - df_["onset"]
        df_["pred"] = df_["pred"].replace(state_name_dict)
        subj_toy_dict = (
            df_.groupby(["pred", "toys"])["duration"].sum()
            / df_.groupby(["pred"])["duration"].sum()
        ).to_dict()
        # print(subj_toy_dict)
        for state in state_name_dict.values():
            for toy in toys:
                key = (state, toy)
                if key in subj_toy_dict.keys():
                    each_cond_time_novice_exp[walk_exp_indicator][task][state][
                        toy
                    ].append(subj_toy_dict[key])
                else:
                    each_cond_time_novice_exp[walk_exp_indicator][task][state][
                        toy
                    ].append(0)

# each_cond_time_novice_exp[task][walk_exp_indicator]

#%%
def draw_toy_experience(
    data_dict, toy_list, toy_name_dict, toy_colors_dict, state_name_dict, name, fig_path, indv
):
    plt.style.use("default")
    offset = 5
    # if indv:
    fig = plt.figure(facecolor="white", figsize=((15, 8)))
    # else:
    #     fig = plt.figure(facecolor="white", figsize=((15, 15)))

    # for state_loc, state in enumerate(state_name_dict.values()):
    this_toy_set = []
    positions = []
    for toy_loc, toy in enumerate(toy_list):
        color = toy_colors_dict[toy]
        plt.boxplot(
            x=data_dict[toy],
            positions=[toy_loc * 4 + 12],
            widths=1,
            patch_artist=True,
            boxprops=dict(facecolor="none", edgecolor=color, lw=5),
            capprops=dict(c=color, lw=5),
            whiskerprops=dict(c=color, lw=5),
            flierprops=dict(markerfacecolor="none", markeredgecolor=color),
            medianprops=dict(c=color, lw=5),
        )
        positions.append(toy_loc * 4 + 12)
        this_toy_set.append(toy_name_dict[toy])
    print(this_toy_set)
    plt.xlabel("Toys", fontsize=28)
    # numpy.linspace(start, stop, num=50)
    plt.xticks(positions, this_toy_set, fontsize=26)
    plt.ylim(top=1.1)
    plt.grid(b = True, axis = 'y', color='black', linestyle='-', linewidth=2)
    plt.yticks(
        np.arange(0, 1.1, 0.1), [str(i) for i in np.arange(0, 110, 10)], fontsize=28
    )
    plt.ylabel("% of total time in session", fontsize=28)
    plt.ylim(bottom = 0, top = 1)

    plt.title(name, fontsize=28)
    if indv:
        handles = []
        for toy in toy_list:
            color = toy_colors_dict[toy]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    c=color,
                    label=toy_name_dict[toy],
                    markerfacecolor=color,
                    markersize=15,
                    fillstyle="full",
                )
            )
        plt.legend(fontsize=20, loc="upper right", handles=handles)
    for x_pos in np.arange(10, len(toys) * 4 + 12, 4):
        plt.axvline(x=x_pos, c="black")

    plt.tight_layout()
    plt.savefig(fig_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("seaborn")


def draw_toy_exp(
    data_dict,
    toy_list,
    toy_colors_dict,
    state_name_dict,
    name,
    fig_path,
    indv,
):
    offset = 35
    if indv:
        fig = plt.figure(facecolor="white", figsize=((15, 8)))
    else:
        fig = plt.figure(facecolor="white", figsize=((15, 15)))

    for state_loc, state in enumerate(state_name_dict.values()):
        for toy_loc, toy in enumerate(toy_list):
            color = toy_colors_dict[toy]
            plt.boxplot(
                x=data_dict[state][toy],
                positions=[state_loc * offset + toy_loc * 4],
                widths=3,
                patch_artist=True,
                boxprops=dict(facecolor="none", edgecolor=color),
                capprops=dict(c=color, lw=3),
                whiskerprops=dict(c=color, lw=2),
                flierprops=dict(markerfacecolor="none", markeredgecolor=color),
                medianprops=dict(c=color, lw=3),
            )
    plt.xlabel("States", fontsize=28)
    plt.xticks([10, 45, 80, 115, 150], list(state_name_dict.values()), fontsize=28)
    plt.ylim(top=1.1)
    plt.yticks(
        np.arange(0, 1.1, 0.1), [str(i) for i in np.arange(0, 110, 10)], fontsize=28
    )
    plt.ylabel("% of total time in that state", fontsize=28)
    plt.title(name, fontsize=28)
    plt.grid(False)
    if not indv:
        handles = []
        for toy in toy_list:
            color = toy_colors_dict[toy]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    c=color,
                    label=toy,
                    markerfacecolor=color,
                    markersize=15,
                    fillstyle="full",
                )
            )
        plt.legend(fontsize=20, loc="upper right", handles=handles)
    plt.tight_layout()
    plt.savefig(fig_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


# %%
name_exp = {"novice": "Novice", "exp": "Experienced"}
name_conditions = {
    "MPS": "with caregiver, fine-motor toys",
    "MPM": "with caregiver, gross-motor toys",
    "NMS": "without caregiver, fine-motor toys",
    "NMM": "without caregiver, gross-motor toys",
}

for exp in ["novice", "exp"]:
    for task in tasks:
        data_dict = each_cond_time_novice_exp[exp][task]
        if task in ["MPS", "NMS"]:
            toys = stationary_toys_list
        elif task in ["MPM", "NMM"]:
            toys = mobile_toys_list
        name = name_exp[exp] + " walkers, " + name_conditions[task]
        fig_path = (
            "../figures/hmm/20210907/"
            + feature_set
            + "/no_ops_threshold_"
            + str(no_ops_time)
            + "/window_size_"
            + str(interval_length)
            + "/"
            + str(n_states)
            + "_states"
            + "/toy_exp_"
            + exp
            + "_"
            + task
            + ".png"
        )

        draw_toy_exp(
            data_dict,
            toys,
            toy_colors_dict,
            state_name_dict,
            name=name,
            fig_path=fig_path,
            indv=True,
        )
    stationary_dict = {}
    toys = stationary_toys_list

    for state in state_name_dict.values():
        stationary_dict[state] = {}
        for toy in toys:
            stationary_dict[state][toy] = (
                each_cond_time_novice_exp[exp]["MPS"][state][toy]
                + each_cond_time_novice_exp[exp]["NMS"][state][toy]
            )

    data_dict = stationary_dict
    name = name_exp[exp] + " walkers, fine-motor toys, both conditions"
    fig_path = (
        "../figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states"
        + "/toy_exp_"
        + exp
        + "_stationary"
        + ".png"
    )

    draw_toy_exp(
        data_dict,
        toys,
        toy_colors_dict,
        state_name_dict,
        name=name,
        fig_path=fig_path,
        indv=False,
    )

    mobile_dict = {}
    toys = mobile_toys_list

    for state in state_name_dict.values():
        mobile_dict[state] = {}
        for toy in toys:
            mobile_dict[state][toy] = (
                each_cond_time_novice_exp[exp]["MPM"][state][toy]
                + each_cond_time_novice_exp[exp]["NMM"][state][toy]
            )

    data_dict = mobile_dict
    name = name_exp[exp] + " walkers, gross-motor toys, both conditions"
    fig_path = (
        "../figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states"
        + "/toy_exp_"
        + exp
        + "_mobile"
        + ".png"
    )

    draw_toy_exp(
        data_dict,
        toys,
        toy_colors_dict,
        state_name_dict,
        name=name,
        fig_path=fig_path,
        indv=False,
    )

#%%
## without the states
each_cond_toy_time_walking_exp = {"novice": {}, "exp": {}}
for task in tasks:
    each_cond_toy_time_walking_exp["novice"][task] = {}
    each_cond_toy_time_walking_exp["exp"][task] = {}
    if task in ["MPS", "NMS"]:
        toys = stationary_toys_list
    elif task in ["MPM", "NMM"]:
        toys = mobile_toys_list
    for toy in toys:
        each_cond_toy_time_walking_exp["novice"][task][toy] = []
        each_cond_toy_time_walking_exp["exp"][task][toy] = []

for task in tasks:
    if task in ["MPS", "NMS"]:
        toys = stationary_toys_list
    elif task in ["MPM", "NMM"]:
        toys = mobile_toys_list
    for subj in subj_list:
        if infant_info["walking_exp"][subj] >= 90:
            walk_exp_indicator = "exp"
        else:
            walk_exp_indicator = "novice"

        df_ = toy_pred_list[task][subj].copy()
        df_ = df_.explode("toys")
        df_["toys"] = df_["toys"].replace({"no_ops": "no_toy"})
        df_["duration"] = df_["offset"] - df_["onset"]
        df_["pred"] = df_["pred"].replace(state_name_dict)
        subj_toy_dict = (
            df_.groupby(["toys"])["duration"].sum() / df_["duration"].sum()
        ).to_dict()
        # print(subj_toy_dict)
        for toy in toys:
            if toy in subj_toy_dict.keys():
                each_cond_toy_time_walking_exp[walk_exp_indicator][task][toy].append(
                    subj_toy_dict[toy]
                )
            else:
                each_cond_toy_time_walking_exp[walk_exp_indicator][task][toy].append(0)

#%%
subj_toy_dict
# %%
name_exp = {"novice": "Novice", "exp": "Experienced"}
name_conditions = {
    "MPS": "with caregiver, fine-motor toys",
    "MPM": "with caregiver, gross-motor toys",
    "NMS": "without caregiver, fine-motor toys",
    "NMM": "without caregiver, gross-motor toys",
}
#%%
toy_name_dict = {'bricks':'Bricks', 'pig':'Pig', 'popuppals':'Pop-up pals',\
                'xylophone': "Xylophone",\
                'shape_sorter': "Shape\nsorter",\
                'shape_sorter_blocks': 'Shape sorter\nblocks',\
                'broom':"Broom",\
                'clear_ball': "Clear Ball",\
                'balls': "Balls",\
                'food': "Food",\
                'grocery_cart': "Grocery Cart",\
                'stroller':"Stroller",\
                'bucket': "Bucket"}
for exp in ["novice", "exp"]:
    for task in tasks:
        data_dict = each_cond_toy_time_walking_exp[exp][task]
        if task in ["MPS", "NMS"]:
            toys = stationary_toys_list
        elif task in ["MPM", "NMM"]:
            toys = mobile_toys_list

        name = name_exp[exp] + " walkers, " + name_conditions[task]
        fig_path = (
            "../figures/hmm/20210907/"
            + feature_set
            + "/no_ops_threshold_"
            + str(no_ops_time)
            + "/window_size_"
            + str(interval_length)
            + "/"
            + str(n_states)
            + "_states"
            + "/toy_exp_"
            + exp
            + "_"
            + task
            + "all_states"
            + ".png"
        )

        draw_toy_experience(
            data_dict,
            toys,
            toy_name_dict, 
            toy_colors_dict,
            state_name_dict,
            name=name,
            fig_path=fig_path,
            indv=True,
        )
    stationary_dict_all_states = {}
    toys = stationary_toys_list

    for toy in toys:
        stationary_dict_all_states[toy] = (
            each_cond_toy_time_walking_exp[exp]["MPS"][toy]
            + each_cond_toy_time_walking_exp[exp]["NMS"][toy]
        )

    data_dict = stationary_dict_all_states
    name = name_exp[exp] + " walkers, fine-motor toys, both conditions"
    fig_path = (
        "../figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states"
        + "/toy_exp_"
        + exp
        + "_stationary_all_states"
        + ".png"
    )

    draw_toy_experience(
        data_dict,
        toys,
        toy_name_dict,
        toy_colors_dict,
        state_name_dict,
        name=name,
        fig_path=fig_path,
        indv=False,
    )

    mobile_dict_all_states = {}
    toys = mobile_toys_list

    for toy in toys:
        mobile_dict_all_states[toy] = (
            each_cond_toy_time_walking_exp[exp]["MPM"][toy]
            + each_cond_toy_time_walking_exp[exp]["NMM"][toy]
        )

    data_dict = mobile_dict_all_states
    name = name_exp[exp] + " walkers, gross-motor toys, both conditions"
    fig_path = (
        "../figures/hmm/20210907/"
        + feature_set
        + "/no_ops_threshold_"
        + str(no_ops_time)
        + "/window_size_"
        + str(interval_length)
        + "/"
        + str(n_states)
        + "_states"
        + "/toy_exp_"
        + exp
        + "_mobile_all_states"
        + ".png"
    )

    draw_toy_experience(
        data_dict,
        toys,
        toy_name_dict,
        toy_colors_dict,
        state_name_dict,
        name=name,
        fig_path=fig_path,
        indv=False,
    )
# %%
# Test difference 
from statsmodels.stats.weightstats import ztest

print("Fine-motor toy")
for toy in stationary_toys_list:
    print(toy)
    print(
        ztest(
        each_cond_toy_time_walking_exp["exp"]["MPS"][toy] + each_cond_toy_time_walking_exp["exp"]["NMS"][toy], 
        each_cond_toy_time_walking_exp["novice"]["MPS"][toy] + each_cond_toy_time_walking_exp["novice"]["NMS"][toy], 
        alternative="larger",
        )
    )

print("\nGross-motor toy")
for toy in mobile_toys_list:
    print(toy)
    print(
        ztest(
        each_cond_toy_time_walking_exp["exp"]["MPM"][toy] + each_cond_toy_time_walking_exp["exp"]["NMM"][toy], 
        each_cond_toy_time_walking_exp["novice"]["MPM"][toy] + each_cond_toy_time_walking_exp["novice"]["NMM"][toy], 
        alternative="larger",
        )
    )
# %%
