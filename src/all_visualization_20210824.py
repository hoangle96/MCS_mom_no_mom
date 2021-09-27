import numpy as np
import pandas as pd
from variables import (
    tasks,
    condition_name,
    state_color_dict,
    stationary_toys_list,
    mobile_toys_list,
    state_color_dict_shades,
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


def get_longest_item(dictionary):
    return max((len(v)) for _, v in dictionary.items())


def convert_to_list_seqs(big_seq, len_array):
    big_seq_to_slice = big_seq.copy()
    list_of_seqs = []

    for k in len_array:
        list_of_seqs.append(big_seq_to_slice[:k])
        big_seq_to_slice = big_seq_to_slice[k:]
    return list_of_seqs


if __name__ == "__main__":
    shift = 0.5
    mobile_toys_list.append("no_toy")
    stationary_toys_list.append("no_toy")

    for feature_set in [
        "n_new_toy_ratio"
    ]:  # , 'fav_toy_till_now', 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio']:
        # for feature_set in ['new_toy_play_time_ratio']:
        for no_ops_time in [10, 7, 5]:
            with open(
                "./data/interim/20210824_"
                + str(no_ops_time)
                + "_no_ops_threshold_clean_data_for_feature_engineering.pickle",
                "rb",
            ) as f:
                task_to_storing_dict = pickle.load(f)

            print("no_ops_time", no_ops_time)
            for interval_length in [1.5, 2, 1]:
                # if (no_ops_time == 5 and interval_length > 1) or no_ops_time != 5:
                # print('interval_length', interval_length)
                shift_time_list = np.arange(0, interval_length, shift)

                with open(
                    "./data/interim/20210907_"
                    + str(no_ops_time)
                    + "_no_ops_threshold_feature_dict_with_"
                    + feature_set
                    + "_"
                    + str(interval_length)
                    + "_min.pickle",
                    "rb",
                ) as f:
                    feature_dict = pickle.load(f)

                with open(
                    "./data/interim/20210907_"
                    + str(no_ops_time)
                    + "_no_ops_threshold_feature_engineering_time_arr_"
                    + str(interval_length)
                    + "_min.pickle",
                    "rb",
                ) as f:
                    time_arr_dict = pickle.load(f)

                with open(
                    "./data/interim/20210907_"
                    + str(no_ops_time)
                    + "_no_ops_threshold_label_"
                    + str(interval_length)
                    + "_min.pickle",
                    "rb",
                ) as f:
                    labels_dict = pickle.load(f)

                with open(
                    "./data/interim/20210907_"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold_discretized_input_list_"
                    + str(interval_length)
                    + "_min.pickle",
                    "rb",
                ) as f:
                    discretized_input_list = pickle.load(f)

                for n_states in [5]:  # range(4, 7):
                    print("states", n_states)
                    with open(
                        "./data/interim/20210907"
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

                    with open(
                        "./data/interim/20210907"
                        + feature_set
                        + "_"
                        + str(no_ops_time)
                        + "_no_ops_threshold_"
                        + str(n_states)
                        + "_states_prediction_all_prob_"
                        + str(interval_length)
                        + "_min.pickle",
                        "rb",
                    ) as f:
                        all_proba_dict = pickle.load(f)

                    with open(
                        "./data/interim/20210907"
                        + feature_set
                        + "_"
                        + str(no_ops_time)
                        + "_no_ops_threshold"
                        + str(n_states)
                        + "_states_merged_prediction_"
                        + str(interval_length)
                        + "_min.pickle",
                        "rb",
                    ) as f:
                        merged_pred_dict_all = pickle.load(f)

                    with open(
                        "./data/interim/20210907"
                        + feature_set
                        + "_"
                        + str(no_ops_time)
                        + "_no_ops_threshold"
                        + str(n_states)
                        + "_states_merged_prediction_prob_"
                        + str(interval_length)
                        + "_min.pickle",
                        "rb",
                    ) as f:
                        merged_proba_dict_all = pickle.load(f)

                    with open(
                        "./data/interim/20210907"
                        + feature_set
                        + "_"
                        + str(no_ops_time)
                        + "_no_ops_threshold"
                        + str(n_states)
                        + "_states_time_arr_dict_"
                        + str(interval_length)
                        + "_min.pickle",
                        "rb",
                    ) as f:
                        time_subj_dict_all = pickle.load(f)

                    with open(
                        "./data/interim/20210907"
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

                    with open(
                        "./data/interim/20210907"
                        + feature_set
                        + "_"
                        + str(no_ops_time)
                        + "_no_ops_threshold"
                        + str(n_states)
                        + "_states_merged_locomotion_"
                        + str(interval_length)
                        + "_min.pickle",
                        "rb",
                    ) as f:
                        merged_pred_w_locomotion = pickle.load(f)

                    # with open('./data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_locomotion_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    # merged_pred_w_locomotion = pickle.load(f)

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
                    model_file_path = (
                        Path("./models/hmm/20210907/" + feature_set) / model_file_name
                    )
                    with open(model_file_path, "rb") as f:
                        model = pickle.load(f)
                    state_name_dict = rank_state(model)

                    subj_list = list(task_to_storing_dict["MPS"].keys())

                    toy_colors_dict = {
                        "bricks": "blue",
                        "pig": "orange",
                        "popuppals": "green",
                        "xylophone": "red",
                        "shape_sorter": "skyblue",
                        "shape_sorter_blocks": "salmon",
                        "broom": "purple",
                        "clear_ball": "teal",
                        "balls": "cadetblue",
                        "food": "chocolate",
                        "grocery_cart": "dodgerblue",
                        "stroller": "violet",
                        "bucket": "navy",
                        "no_toy": "slategrey",
                    }

                    ### state pred figures

                    if (
                        feature_set == "n_new_toy_ratio"
                        or feature_set == "n_new_toy_ratio_and_fav_toy_till_now"
                    ):
                        x_ticks_dict = {
                            0: ["[0, 4)", "[4, 8)", "[8, 12)", "[12+"],
                            1: ["0", "1", "2", "3", "4+"],
                            2: [
                                "[0, .2)",
                                "[.2, .4)",
                                "[.4, .6)",
                                "[.6, .8)",
                                "[.8, 1]",
                            ],
                            3: [
                                "[0, .2)",
                                "[.2, .4)",
                                "[.4, .6)",
                                "[.6, .8)",
                                "[.8, 1]",
                            ],
                        }
                        feature_names = [
                            "# toys switches",
                            "# toys",
                            "# new toys ratio",
                            "fav toy ratio",
                        ]
                        feature_values = {
                            0: range(1, 5),
                            1: range(5),
                            2: range(1, 6),
                            3: range(1, 6),
                        }

                    elif feature_set == "fav_toy_till_now":
                        x_ticks_dict = {
                            0: ["[0, 4)", "[4, 8)", "[8, 12)", "[12+"],
                            1: ["0", "1", "2", "3", "4+"],
                            2: ["0", "1", "2", "3", "4+"],
                            3: [
                                "[0, .2)",
                                "[.2, .4)",
                                "[.4, .6)",
                                "[.6, .8)",
                                "[.8, 1]",
                            ],
                        }
                        feature_names = [
                            "# toys switches",
                            "# toys",
                            "# new toys",
                            "fav toy ratio",
                        ]
                        feature_values = {
                            0: range(1, 5),
                            1: range(5),
                            2: range(5),
                            3: range(1, 6),
                        }
                    elif feature_set == "new_toy_play_time_ratio":
                        x_ticks_dict = {
                            0: ["[0, 4)", "[4, 8)", "[8, 12)", "[12+"],
                            1: ["0", "1", "2", "3", "4+"],
                            2: [
                                "[0, .2)",
                                "[.2, .4)",
                                "[.4, .6)",
                                "[.6, .8)",
                                "[.8, 1]",
                            ],
                            3: [
                                "[0, .2)",
                                "[.2, .4)",
                                "[.4, .6)",
                                "[.6, .8)",
                                "[.8, 1]",
                            ],
                        }
                        feature_names = [
                            "# toys switches",
                            "# toys",
                            "new toys play time ratio",
                            "fav toy ratio",
                        ]
                        feature_values = {
                            0: range(1, 5),
                            1: range(5),
                            2: range(1, 6),
                            3: range(1, 6),
                        }

                    n_features = 4
                    flatten_pred = []
                    flatten_pred_dict = {}
                    discretized_input_list_by_task = {}
                    discretized_input = discretized_input_list.copy()
                    len_dict = {}
                    for task in tasks:
                        len_list = []
                        for subj, shifted_df_dict in feature_dict[task].items():
                            for shift_time, feature_vector in shifted_df_dict.items():
                                # if feature_set == 'n_new_toy_ratio_and_fav_toy_till_now':
                                # m, n, _ = feature_vector.shape
                                # feature_vector = feature_vector.reshape((n, m))
                                # print(feature_vector.shape)

                                # input_list = np.vstack((input_list, feature_vector))
                                len_list.append(len(feature_vector))
                        len_dict[task] = len_list

                    len_array = [sum(i) for i in len_dict.values()]
                    discretized_input_list_each_task = convert_to_list_seqs(
                        discretized_input, len_array
                    )
                    Path(
                        "./figures/hmm/20210907/"
                        + feature_set
                        + "/no_ops_threshold_"
                        + str(no_ops_time)
                        + "/window_size_"
                        + str(interval_length)
                        + "/"
                        + str(n_states)
                        + "_states/"
                    ).mkdir(parents=True, exist_ok=True)
                    for idx, task in enumerate(tasks):
                        flatten_pred_dict[task] = []
                        task_specific_pred_dict = pred_dict[task]
                        for subj, subj_dict in task_specific_pred_dict.items():
                            for shift_time, pred in subj_dict.items():
                                flatten_pred.extend(pred)
                                flatten_pred_dict[task].extend(pred)
                                # discretized_input_list_by_task

                        fig_path = (
                            "./figures/hmm/20210907/"
                            + feature_set
                            + "/no_ops_threshold_"
                            + str(no_ops_time)
                            + "/window_size_"
                            + str(interval_length)
                            + "/"
                            + str(n_states)
                            + "_states/distribution_time_in_state_"
                            + task
                            + ".png"
                        )
                        draw_state_distribution(
                            flatten_pred_dict[task],
                            n_states,
                            state_name_dict,
                            condition_name[task],
                            state_color_dict_shades,
                            fig_path,
                        )

                        fig_path = (
                            "./figures/hmm/20210907/"
                            + feature_set
                            + "/no_ops_threshold_"
                            + str(no_ops_time)
                            + "/window_size_"
                            + str(interval_length)
                            + "/"
                            + str(n_states)
                            + "_states/emission_distribution_"
                            + task
                            + ".png"
                        )
                        draw_distribution(
                            n_features,
                            state_name_dict,
                            discretized_input_list_each_task[idx],
                            np.array(flatten_pred_dict[task]),
                            "",
                            feature_names,
                            x_ticks_dict,
                            feature_values,
                            state_color_dict_shades,
                            fig_path,
                        )

                    # fig_path = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/state_distribution.png"
                    # draw_state_distribution(flatten_pred, n_states, state_name_dict, "Distribution of time spent in each state, " +'\n'+ str(no_ops_time) + 's threshold,window size ' + str(interval_length) +" min", state_color_dict_shades, fig_path)

                    # fig_path = './figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/emission_distribution.png"
                    # draw_distribution(n_features, state_name_dict, discretized_input_list, np.array(flatten_pred), str(no_ops_time) + 's threshold, window size ' + str(interval_length) +" min",feature_names, x_ticks_dict, feature_values, state_color_dict_shades, fig_path)

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

                        (
                            merged_pred_dict_all_task_specific,
                            merged_proba_dict_all_task_specific,
                            time_subj_dict_all_task_specific,
                            all_prob,
                        ) = merge_segment_with_state_calculation_all(
                            subj_list,
                            shift_time_list,
                            merged_df_dict,
                            time_arr_shift_dict,
                            pred_subj_dict,
                            prob_subj_dict,
                            window_size=interval_length,
                            n_states=n_states,
                            shift_interval=60000 * shift,
                        )

                        merged_pred_dict_all[task] = merged_pred_dict_all_task_specific
                        merged_proba_dict_all[
                            task
                        ] = merged_proba_dict_all_task_specific
                        time_subj_dict_all[task] = time_subj_dict_all_task_specific
                        all_prob_dict_all[task] = all_prob

                    for subj in subj_list:
                        for task in tasks:
                            path = Path(
                                "./figures/hmm/20210907/"
                                + feature_set
                                + "/no_ops_threshold_"
                                + str(no_ops_time)
                                + "/window_size_"
                                + str(interval_length)
                                + "/"
                                + str(n_states)
                                + "_states/merged/"
                                + task
                                + "/"
                            )
                            path.mkdir(parents=True, exist_ok=True)
                            df = pd.DataFrame()
                            for df_ in task_to_storing_dict[task][subj]:
                                df = pd.concat([df, df_])
                            pred_state_list = merged_pred_dict_all[task][subj]
                            state_name_list = [
                                state_name_dict[s] for s in pred_state_list
                            ]
                            time_list = time_subj_dict_all[task][subj]
                            prob_list = all_prob_dict_all[task][subj]
                            fig_name = (
                                "./figures/hmm/20210907/"
                                + feature_set
                                + "/no_ops_threshold_"
                                + str(no_ops_time)
                                + "/window_size_"
                                + str(interval_length)
                                + "/"
                                + str(n_states)
                                + "_states/merged/"
                                + task
                                + "/"
                                + str(subj)
                                + ".png"
                            )
                            draw_timeline_with_prob_to_check(
                                k=str(subj)
                                + "window size: "
                                + str(interval_length)
                                + " no ops threshold "
                                + str(no_ops_time),
                                df=df,
                                state_list=state_name_list,
                                time_list=time_list,
                                state_name=state_name_dict,
                                fig_name=fig_name,
                                gap_size=shift,
                                state_color_dict=state_color_dict_shades,
                                prob_list=prob_list,
                                shift=shift,
                            )

                    # with opsimien('./data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    # with open('./data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_all_pred_prob_'+str(interval_length)+'_min.pickle', 'wb+') as f:
                    #     pickle.dump(all_prob_dict_all, f)
                    #     # print(pred_dict)
                    for subj in subj_list:
                        for task in tasks:
                            for shift_time in np.arange(0, interval_length, shift):
                                if shift_time in [0.0, 1.0, 2.0]:
                                    shift_time = int(shift_time)
                                path = Path(
                                    "./figures/hmm/20210824/"
                                    + feature_set
                                    + "/no_ops_threshold_"
                                    + str(no_ops_time)
                                    + "/window_size_"
                                    + str(interval_length)
                                    + "/"
                                    + str(n_states)
                                    + "_states/shift_"
                                    + str(shift_time)
                                    + "/"
                                    + task
                                    + "/"
                                )
                                path.mkdir(parents=True, exist_ok=True)

                                df = pd.DataFrame()
                                for df_ in task_to_storing_dict[task][subj]:
                                    df = pd.concat([df, df_])
                                pred_state_list = pred_dict[task][subj][shift_time]
                                # print(pred_state_list)
                                state_name_list = [
                                    state_name_dict[s] for s in pred_state_list
                                ]
                                time_list = time_arr_dict[task][subj][shift_time]
                                prob_list = all_proba_dict[task][subj][shift_time]
                                # print(prob_list)
                                # print(state_name_list)

                                if len(time_list) < 2:
                                    print(subj, task, shift_time)

                                if len(time_list) != len(prob_list):
                                    print(subj, task, shift_time)
                                else:
                                    fig_name = (
                                        "./figures/hmm/20210907/"
                                        + feature_set
                                        + "/no_ops_threshold_"
                                        + str(no_ops_time)
                                        + "/window_size_"
                                        + str(interval_length)
                                        + "/"
                                        + str(n_states)
                                        + "_states/shift_"
                                        + str(shift_time)
                                        + "/"
                                        + task
                                        + "/"
                                        + str(subj)
                                        + ".png"
                                    )
                                    # draw_timeline_with_prob_to_check(k = subj, df = df, state_list = pred_state_list, time_list = time_list, \
                                    #                                 state_name = state_name_dict, state_color_dict = state_color_dict,\
                                    #                                 fig_name= fig_name, gap_size = interval_length, prob_list = prob_list, shift = shift)
                                    # draw_timeline_with_merged_states(str(subj) + "window size: " + str(interval_length) + " no ops threshold "+ str(no_ops_time), df, pred_state_list, time_list, state_name_dict, fig_name= fig_name, gap_size = interval_length, state_color_dict= state_color_dict_shades)
                                    draw_timeline_with_prob_to_check(
                                        k=str(subj)
                                        + " window size: "
                                        + str(interval_length)
                                        + " no ops threshold "
                                        + str(no_ops_time)
                                        + " shift time: "
                                        + str(shift_time),
                                        df=df,
                                        state_list=state_name_list,
                                        time_list=time_list,
                                        state_name=state_name_dict,
                                        fig_name=fig_name,
                                        gap_size=interval_length,
                                        state_color_dict=state_color_dict_shades,
                                        prob_list=prob_list,
                                        shift=shift,
                                    )

                    # # toy state figures
                    stationary_dict_for_std = {}
                    for state in state_name_dict.values():
                        stationary_dict_for_std[state] = {}
                        for toy in stationary_toys_list:
                            stationary_dict_for_std[state][toy] = []

                    stationary_df = pd.DataFrame()
                    for task in ["MPS", "NMS"]:
                        for subj in subj_list:
                            if subj in toy_pred_list[task].keys():
                                df_ = toy_pred_list[task][subj].copy()
                                df_ = df_.explode("toys")
                                df_["toys"] = df_["toys"].replace({"no_ops": "no_toy"})

                                df_["duration"] = df_["offset"] - df_["onset"]
                                df_["pred"] = df_["pred"].replace(state_name_dict)
                                subj_stationary_dict = (
                                    df_.groupby(["pred", "toys"])["duration"].sum()
                                    / df_.groupby(["pred"])["duration"].sum()
                                ).to_dict()
                                for state in state_name_dict.values():
                                    for toy in stationary_toys_list:
                                        key = (state, toy)
                                        if key in subj_stationary_dict.keys():
                                            stationary_dict_for_std[state][toy].append(
                                                subj_stationary_dict[key]
                                            )
                                        else:
                                            stationary_dict_for_std[state][toy].append(
                                                0
                                            )

                                stationary_df = pd.concat(
                                    [stationary_df, toy_pred_list[task][subj]]
                                )
                    stationary_df = stationary_df.explode("toys")
                    stationary_df["toys"] = stationary_df["toys"].replace(
                        {"no_ops": "no_toy"}
                    )

                    stationary_df["duration"] = (
                        stationary_df["offset"] - stationary_df["onset"]
                    )
                    stationary_df["pred"] = stationary_df["pred"].replace(
                        state_name_dict
                    )
                    stationary_toy_to_pred_dict = (
                        stationary_df.groupby(["pred", "toys"])["duration"].sum()
                        / stationary_df.groupby(["pred"])["duration"].sum()
                    ).to_dict()
                    stationary_toy_list = stationary_df["toys"].dropna().unique()

                    stationary_std = {}
                    stationary_median = {}
                    for state in state_name_dict.values():
                        stationary_std[state] = {}
                        for toy in stationary_toys_list:
                            key = (state, toy)
                            if key in stationary_toy_to_pred_dict.keys():
                                stationary_median[key] = np.mean(
                                    stationary_dict_for_std[state][toy]
                                )
                                stationary_std[state][toy] = np.std(
                                    stationary_dict_for_std[state][toy]
                                )

                                # stationary_std[state][toy] = np.abs(np.sum(np.array(stationary_dict_for_std[state][toy])-stationary_toy_to_pred_dict[key]))/len(stationary_dict_for_std[state][toy])

                    name = "Both conditions, fine motor toys"
                    # fig_path = './figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/toy_state_stationary.png'
                    # draw_toy_state(state_name_dict, stationary_toy_to_pred_dict, toy_colors_dict= toy_colors_dict, toy_list =  stationary_toy_list, name = name, fig_path =  fig_path,  indv = False)

                    fig_path = (
                        "./figures/hmm/20210907/"
                        + feature_set
                        + "/no_ops_threshold_"
                        + str(no_ops_time)
                        + "/window_size_"
                        + str(interval_length)
                        + "/"
                        + str(n_states)
                        + "_states"
                        + "/toy_state_stationary_w_std.png"
                    )
                    draw_toy_state_with_std(
                        state_name_dict,
                        stationary_median,
                        std_dict=stationary_std,
                        toy_colors_dict=toy_colors_dict,
                        toy_list=stationary_toys_list,
                        name=name,
                        fig_path=fig_path,
                        indv=False,
                    )

                    mobile_df = pd.DataFrame()
                    mobile_dict_for_std = {}

                    for state in state_name_dict.values():
                        mobile_dict_for_std[state] = {}
                        for toy in mobile_toys_list:
                            mobile_dict_for_std[state][toy] = []

                    for task in ["MPM", "NMM"]:
                        for subj in subj_list:
                            if subj in toy_pred_list[task].keys():
                                df_ = toy_pred_list[task][subj].copy()
                                df_ = df_.explode("toys")
                                df_["toys"] = df_["toys"].replace({"no_ops": "no_toy"})

                                df_["duration"] = df_["offset"] - df_["onset"]
                                df_["pred"] = df_["pred"].replace(state_name_dict)
                                subj_mobile_dict = (
                                    df_.groupby(["pred", "toys"])["duration"].sum()
                                    / df_.groupby(["pred"])["duration"].sum()
                                ).to_dict()
                                for state in state_name_dict.values():
                                    for toy in mobile_toys_list:
                                        key = (state, toy)
                                        if key in subj_mobile_dict.keys():
                                            mobile_dict_for_std[state][toy].append(
                                                subj_mobile_dict[key]
                                            )
                                        else:
                                            mobile_dict_for_std[state][toy].append(0)

                                mobile_df = pd.concat(
                                    [mobile_df, toy_pred_list[task][subj]]
                                )
                    mobile_df = mobile_df.explode("toys")
                    mobile_df["toys"] = mobile_df["toys"].replace({"no_ops": "no_toy"})
                    mobile_df["duration"] = mobile_df["offset"] - mobile_df["onset"]
                    mobile_df["pred"] = mobile_df["pred"].replace(state_name_dict)
                    mobile_toy_to_pred_dict = (
                        mobile_df.groupby(["pred", "toys"])["duration"].sum()
                        / mobile_df.groupby(["pred"])["duration"].sum()
                    ).to_dict()
                    mobile_toy_list = list(mobile_df["toys"].dropna().unique())
                    # print(mobile_toy_list)
                    # print(mobile_dict_for_std)
                    mobile_std = {}
                    mobile_median = {}
                    for state in state_name_dict.values():
                        mobile_std[state] = {}
                        for toy in mobile_toy_list:
                            key = (state, toy)
                            # print(key)
                            if key in mobile_toy_to_pred_dict.keys():
                                mobile_median[key] = np.mean(
                                    mobile_dict_for_std[state][toy]
                                )
                                mobile_std[state][toy] = np.std(
                                    mobile_dict_for_std[state][toy]
                                )

                                # mobile_std[state][toy] = np.abs(np.sum(np.array(mobile_dict_for_std[state][toy])-mobile_toy_to_pred_dict[key]))/len(mobile_dict_for_std[state][toy])
                    name = "Both conditions, gross motor toys"
                    # fig_path = './figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/toy_state_mobile.png'
                    # draw_toy_state(state_name_dict, mobile_toy_to_pred_dict, toy_colors_dict= toy_colors_dict, toy_list = mobile_toy_list, name = name, fig_path =  fig_path,  indv = False)
                    fig_path = (
                        "./figures/hmm/20210907/"
                        + feature_set
                        + "/no_ops_threshold_"
                        + str(no_ops_time)
                        + "/window_size_"
                        + str(interval_length)
                        + "/"
                        + str(n_states)
                        + "_states"
                        + "/toy_state_mobile_w_std_20210901.png"
                    )
                    # draw_toy_state_with_std(state_name_dict, mobile_toy_to_pred_dict, toy_colors_dict= toy_colors_dict, toy_list = mobile_toy_list, name = name, fig_path =  fig_path,  indv = False)
                    draw_toy_state_with_std(
                        state_name_dict,
                        mobile_median,
                        toy_colors_dict=toy_colors_dict,
                        toy_list=mobile_toys_list,
                        name=name,
                        fig_path=fig_path,
                        indv=False,
                        std_dict=mobile_std,
                    )

                    fig_name_by_task = {
                        "MPS": "With caregivers, fine motor toys",
                        "MPM": "With caregivers, gross motor toys",
                        "NMS": "Without caregivers, fine motor toys",
                        "NMM": "Without caregivers, gross motor toys",
                    }

                    for task in tasks:
                        if task == "NMM" or task == "MPM":
                            toys_list = mobile_toys_list
                        elif task == "NMS" or task == "MPS":
                            toys_list = stationary_toys_list
                        # print(toys_list)

                        task_dict_for_std = {}
                        for state in state_name_dict.values():
                            task_dict_for_std[state] = {}
                            for toy in toys_list:
                                task_dict_for_std[state][toy] = []

                        df = pd.DataFrame()
                        for subj in subj_list:
                            if subj in toy_pred_list[task].keys():
                                df = pd.concat([df, toy_pred_list[task][subj]])
                                df_ = toy_pred_list[task][subj].copy()
                                df_ = df_.explode("toys")
                                df_["toys"] = df_["toys"].replace({"no_ops": "no_toy"})

                                df_["duration"] = df_["offset"] - df_["onset"]
                                df_["pred"] = df_["pred"].replace(state_name_dict)
                                subj_mobile_dict = (
                                    df_.groupby(["pred", "toys"])["duration"].sum()
                                    / df_.groupby(["pred"])["duration"].sum()
                                ).to_dict()
                                for state in state_name_dict.values():
                                    for toy in toys_list:
                                        key = (state, toy)
                                        if key in subj_mobile_dict.keys():
                                            task_dict_for_std[state][toy].append(
                                                subj_mobile_dict[key]
                                            )
                                        else:
                                            task_dict_for_std[state][toy].append(0)

                        df = df.explode("toys")
                        df["toys"] = df["toys"].replace({"no_ops": "no_toy"})

                        df["duration"] = df["offset"] - df["onset"]
                        df["pred"] = df["pred"].replace(state_name_dict)
                        toy_to_pred_dict = (
                            df.groupby(["pred", "toys"])["duration"].sum()
                            / df.groupby(["pred"])["duration"].sum()
                        ).to_dict()
                        toy_list = np.sort(df["toys"].dropna().unique())
                        name = fig_name_by_task[task]
                        std_dict = {}
                        median_dict = {}
                        for state in state_name_dict.values():
                            std_dict[state] = {}
                            for toy in toys_list:
                                key = (state, toy)
                                if key in toy_to_pred_dict.keys():
                                    median_dict[key] = np.mean(
                                        task_dict_for_std[state][toy]
                                    )
                                    std_dict[state][toy] = np.std(
                                        task_dict_for_std[state][toy]
                                    )
                                    # std_dict[state][toy] = np.abs(np.sum(np.array(task_dict_for_std[state][toy])-toy_to_pred_dict[key]))/len(task_dict_for_std[state][toy])

                        # fig_path = './figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/'+task+'.png'
                        # draw_toy_state(state_name_dict, toy_to_pred_dict = toy_to_pred_dict, toy_list = toy_list, toy_colors_dict = toy_colors_dict, name = name, fig_path= fig_path, indv = True)
                        fig_path = (
                            "./figures/hmm/20210907/"
                            + feature_set
                            + "/no_ops_threshold_"
                            + str(no_ops_time)
                            + "/window_size_"
                            + str(interval_length)
                            + "/"
                            + str(n_states)
                            + "_states"
                            + "/"
                            + task
                            + "_with_std_20210901.png"
                        )
                        draw_toy_state_with_std(
                            state_name_dict,
                            toy_to_pred_dict=median_dict,
                            toy_list=toys_list,
                            toy_colors_dict=toy_colors_dict,
                            name=name,
                            fig_path=fig_path,
                            indv=True,
                            std_dict=std_dict,
                        )

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

                                movement_time_by_each_task_mean_each_infant[task][
                                    state
                                ] = []
                                steps_by_each_task_mean_each_infant[task][state] = []

                        for subj, df in merged_pred_w_locomotion[task].items():
                            df["pred"] = df["pred"].replace(state_name_dict)
                            # print(df['pred'].dtype)
                            # print(df.head())
                            for state in range(n_states):
                                # print(state)
                                df_ = df.loc[df.loc[:, "pred"] == str(state), :]
                                # print(df_.head())

                                if len(df_) > 0:
                                    steps = df_["steps"].to_numpy() * 2
                                    movement_time = (
                                        df_["movement_time"].to_numpy() / 30000
                                    )
                                    steps_mean = np.mean(df_["steps"].to_numpy() * 2)
                                    movement_time_mean = np.mean(
                                        df_["movement_time"].to_numpy() / 30000
                                    )
                                else:
                                    steps = []
                                    movement_time = []
                                    steps_mean = 0
                                    movement_time_mean = 0
                                    steps_by_each_task_for_std[task][state] = None
                                    movement_time_by_each_task_for_std[task][
                                        state
                                    ] = None
                                steps_by_each_task[task][state].extend(steps)
                                movement_time_by_each_task[task][state].extend(
                                    movement_time
                                )

                                movement_time_by_each_task_mean_each_infant[task][
                                    state
                                ].append(movement_time_mean)
                                steps_by_each_task_mean_each_infant[task][state].append(
                                    steps_mean
                                )

                        for state in range(n_states):
                            steps_by_each_task_for_std[task][state] = np.sqrt(
                                np.mean(
                                    np.abs(
                                        movement_time_by_each_task_mean_each_infant[
                                            task
                                        ][state]
                                        - np.mean(steps_by_each_task[task][state])
                                    )
                                    ** 2
                                )
                            )
                            movement_time_by_each_task_for_std[task][state] = np.sqrt(
                                np.mean(
                                    np.abs(
                                        movement_time_by_each_task_mean_each_infant[
                                            task
                                        ][state]
                                        - np.mean(
                                            movement_time_by_each_task[task][state]
                                        )
                                    )
                                    ** 2
                                )
                            )

                    fig_path = (
                        "./figures/hmm/20210907/"
                        + feature_set
                        + "/no_ops_threshold_"
                        + str(no_ops_time)
                        + "/window_size_"
                        + str(interval_length)
                        + "/"
                        + str(n_states)
                        + "_states/step_by_state_with_std_mean.png"
                    )
                    print(np.median(movement_time_by_each_task["MPS"][3]))
                    print(np.median(movement_time_by_each_task["MPS"][2]))
                    draw_mean_state_locotion_across_conditions_separate_mean_std(
                        mean_dict=steps_by_each_task_mean_each_infant,
                        std_dict=steps_by_each_task_for_std,
                        task_list=["MPM", "NMM", "MPS", "NMS"],
                        condition_name=condition_name,
                        n_states=n_states,
                        ylabel="avg # steps/min",
                        title="Avg number of steps in each state for each condition,\n"
                        + str(no_ops_time)
                        + "s threshold, window size "
                        + str(interval_length),
                        figname=fig_path,
                    )

                    fig_path = (
                        "./figures/hmm/20210907/"
                        + feature_set
                        + "/no_ops_threshold_"
                        + str(no_ops_time)
                        + "/window_size_"
                        + str(interval_length)
                        + "/"
                        + str(n_states)
                        + "_states/loco_time_by_state_with_std_mean.png"
                    )

                    draw_mean_state_locotion_across_conditions_separate_mean_std(
                        mean_dict=movement_time_by_each_task_mean_each_infant,
                        std_dict=movement_time_by_each_task_for_std,
                        task_list=["MPM", "NMM", "MPS", "NMS"],
                        condition_name=condition_name,
                        n_states=n_states,
                        ylabel="portion of time in state moving",
                        title="Portion of time in motion in each state for each condition,\n"
                        + str(no_ops_time)
                        + "s threshold, window size "
                        + str(interval_length),
                        figname=fig_path,
                    )
                    # n_infants state per min
                    if n_states == 5:
                        cnt_dict_task_specific = {}
                        for task in tasks:
                            cnt_dict_task_specific[task] = {}
                            len_ = get_longest_item(merged_pred_dict_all[task])
                            for i in range(n_states):
                                cnt_dict_task_specific[task][str(i)] = [0] * len_

                        for task in tasks:
                            for subj, state_list in merged_pred_dict_all[task].items():
                                for state_key, state_name in state_name_dict.items():
                                    # state_list = np.where(state_list == state_key, state_name, state_list)
                                    named_state_list = [
                                        state_name_dict[s] for s in state_list
                                    ]
                                # print(state_list)
                                # print(named_state_list)

                                for idx, state in enumerate(named_state_list):
                                    cnt_dict_task_specific[task][state][idx] += 1

                            focus_state = np.array(
                                cnt_dict_task_specific[task]["1"]
                            ) + np.array(cnt_dict_task_specific[task]["2"])
                            explore_state = np.array(
                                cnt_dict_task_specific[task]["3"]
                            ) + np.array(cnt_dict_task_specific[task]["4"])
                            file_name = (
                                "./figures/hmm/20210907/"
                                + feature_set
                                + "/no_ops_threshold_"
                                + str(no_ops_time)
                                + "/window_size_"
                                + str(interval_length)
                                + "/"
                                + str(n_states)
                                + "_states"
                                + "/"
                                + "n_infants_each_state_per_min_"
                                + task
                                + ".png"
                            )
                            draw_infant_each_min_matplotlib(
                                focus_state,
                                explore_state,
                                cnt_dict_task_specific[task]["0"],
                                condition_name[task],
                                file_name,
                            )
