import pickle
import numpy as np
import pomegranate as pom

# import sys
# sys.path.append('../src')
from visualization import draw_timeline_with_merged_states
from hmm_model import (
    convert_to_int,
    convert_to_list_seqs,
    save_csv,
    init_hmm,
    rank_state,
)

from variables import toys_dict, tasks, toys_list
import matplotlib.pyplot as plt

from merge import merge_segment_with_state_calculation_all, merge_toy_pred
import pandas as pd

from pathlib import Path
import importlib
from collections import OrderedDict

# from all_visualization_20210824 import rank_state
from merge_with_locomotion import merge_movement

shift = 0.5
with open("./data/interim/20210718_babymovement.pickle", "rb") as f:
    babymovement_dict = pickle.load(f)

for feature_set in [
    "n_new_toy_ratio",
    "n_new_toy_ratio_and_fav_toy_till_now",
    "new_toy_play_time_ratio",
    "new_toy_play_time_ratio",
    "fav_toy_till_now",
]:
    # for feature_set in ['new_toy_play_time_ratio']:
    print(feature_set)
    for no_ops_time in [10, 5, 7]:
        print("no_ops_time", no_ops_time)
        for interval_length in [1.5, 2, 1]:
            print("interval_length", interval_length)

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
                "./data/interim/20210824_"
                + str(no_ops_time)
                + "_no_ops_threshold_clean_data_for_feature_engineering.pickle",
                "rb",
            ) as f:
                task_to_storing_dict = pickle.load(f)

            n_features = 4
            shift_time_list = np.arange(0, interval_length, shift)

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
                        len_list.append(len(feature_vector))

            all_labels = []
            for task in tasks:
                for subj, shifted_sequence in labels_dict[task].items():
                    for shift_time, label in shifted_sequence.items():
                        all_labels.append(label)

            # Discretize features
            toy_switch_bins = [0, 5, 10, 15]
            n_bin_ep_rate = range(len(toy_switch_bins))
            discretized_toy_switch_rate = np.digitize(
                input_list[:, 0], toy_switch_bins, right=False
            )
            discretized_n_toys = np.where(input_list[:, 1] > 4, 4, input_list[:, 1])

            if (
                feature_set == "n_new_toy_ratio"
                or feature_set == "n_new_toy_ratio_and_fav_toy_till_now"
                or feature_set == "new_toy_play_time_ratio"
            ):
                new_toys_bin = [0, 0.2, 0.4, 0.6, 0.8]
                discretized_n_new_toys = np.digitize(
                    input_list[:, 2].copy(), new_toys_bin, right=False
                )
            elif feature_set == "fav_toy_till_now":
                discretized_n_new_toys = np.where(
                    input_list[:, 2] > 4, 4, input_list[:, 2]
                )

            # print(np.unique(discretized_n_new_toys))
            # print(np.unique(input_list[:,2]))

            fav_toy_bin = [0, 0.2, 0.4, 0.6, 0.8]

            fav_toy_rate_discretized = np.digitize(
                input_list[:, 3].copy(), fav_toy_bin, right=False
            )

            discretized_input_list = np.hstack(
                (
                    discretized_toy_switch_rate.reshape((-1, 1)),
                    discretized_n_toys.reshape((-1, 1)),
                    discretized_n_new_toys.reshape((-1, 1)),
                    fav_toy_rate_discretized.reshape((-1, 1)),
                )
            )

            with open(
                "./data/interim/20210907_"
                + feature_set
                + "_"
                + str(no_ops_time)
                + "_no_ops_threshold_discretized_input_list_"
                + str(interval_length)
                + "_min.pickle",
                "wb+",
            ) as f:
                pickle.dump(discretized_input_list, f)
            list_seq = convert_to_list_seqs(discretized_input_list, len_list)
            list_seq = convert_to_int(list_seq)

            seed = 1
            for n_states in [5]:  # range(4, 7):
                print("n_states", n_states)
                model = init_hmm(n_states, discretized_input_list.T, seed)
                model.bake()

                # freeze the no_toys distribution so that its parameters are not updated.
                # "no_toys" state params are set so that all of the lowest bins = 0
                for s in model.states:
                    if s.name == "no_toys":
                        for p in s.distribution.parameters[0]:
                            p.frozen = True
                model.fit(list_seq, labels=all_labels)

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
                    Path("./models/hmm/20210926/" + feature_set) / model_file_name
                )
                Path("./models/hmm/20210926/" + feature_set).mkdir(
                    parents=True, exist_ok=True
                )
                with open(model_file_path, "wb+") as f:
                    pickle.dump(model, f)

                state_name_dict = rank_state(model)
                print(state_name_dict)
                data = []

                index_list = [[], []]

                features_obs_dict = {
                    0: len(toy_switch_bins),
                    1: 5,
                    2: len(new_toys_bin),
                    3: 5,
                }

                for i in range(n_features):
                    single_list = np.empty((features_obs_dict[i], n_states))
                    for state_idx, state_i in enumerate(range(n_states)):
                        observation_dict = (
                            model.states[state_i]
                            .distribution.parameters[0][i]
                            .parameters[0]
                        )
                        for idx, k in enumerate(observation_dict.keys()):
                            single_list[idx, state_idx] = np.round(
                                observation_dict[k], 2
                            )
                    index_list[0].extend([i] * len(observation_dict.keys()))
                    index_list[1].extend([i for i in observation_dict.keys()])

                    data.extend(single_list)

                tuples = list(zip(*index_list))
                index = pd.MultiIndex.from_tuples(
                    tuples, names=["feature", "observation"]
                )
                df = pd.DataFrame(
                    data,
                    index=index,
                    columns=[
                        "state " + str(state_name_dict[i]) for i in range(n_states)
                    ],
                )
                file_path = Path(
                    "/scratch/mom_no_mom/reports/20210926/"
                    + feature_set
                    + "/no_ops_threshold_"
                    + str(no_ops_time)
                    + "/window_size_"
                    + str(interval_length)
                    + "/state_"
                    + str(n_states)
                )
                file_path.mkdir(parents=True, exist_ok=True)
                file_name = "mean_" + str(n_states) + "_states" + ".csv"
                save_csv(df, file_path, file_name)

                # save the transition matrix for all
                trans_matrix = pd.DataFrame(
                    np.round(
                        model.dense_transition_matrix()[: n_states + 1, :n_states], 2
                    )
                )
                file_name = (
                    "trans_matrix_"
                    + str(n_states)
                    + "_states_seed_"
                    + str(seed)
                    + "_"
                    + str(interval_length)
                    + "_min.csv"
                )
                trans_matrix = trans_matrix.rename(state_name_dict, axis=1)
                index = state_name_dict.copy()
                index[n_states] = "init_prob"
                trans_matrix = trans_matrix.rename(index, axis=0)
                save_csv(trans_matrix, file_path, file_name)

                i = 0
                input_dict = {}
                for task in tasks:
                    if task not in input_dict.keys():
                        input_dict[task] = {}

                    for subj, shifted_df_dict in feature_dict[task].items():
                        if subj not in input_dict[task].keys():
                            input_dict[task][subj] = {}

                        for shift_time, feature_vector in shifted_df_dict.items():
                            input_dict[task][subj][shift_time] = list_seq[i]
                            i += 1

                total_log_prob = 0
                log_prob_list = []
                pred_dict = {}
                proba_dict = {}
                all_proba_dict = {}

                pred_by_task = {}
                input_by_task = {}

                for task in tasks:
                    if task not in pred_dict.keys():
                        pred_dict[task] = {}
                        proba_dict[task] = {}
                        all_proba_dict[task] = {}
                        pred_by_task[task] = []
                        input_by_task[task] = []

                    for subj, shifted_dict in input_dict[task].items():
                        if subj not in pred_dict[task].keys():
                            pred_dict[task][subj] = {}
                            proba_dict[task][subj] = {}
                            all_proba_dict[task][subj] = {}

                        for shift_time, feature_vector in shifted_dict.items():
                            label = model.predict(feature_vector)
                            label = [int(state_name_dict[s]) for s in label]
                            pred_dict[task][subj][shift_time] = label
                            pred_by_task[task].extend(label)
                            input_by_task[task].extend(feature_vector)

                            # if 4 in label:
                            # print(feature_vector, label)
                            proba_dict[task][subj][shift_time] = np.amax(
                                model.predict_proba(feature_vector), axis=1
                            )

                            log_prob = model.log_probability(feature_vector)
                            # print(list(state_name_dict.keys()))
                            all_proba_dict[task][subj][
                                shift_time
                            ] = model.predict_proba(feature_vector)[
                                :, list(state_name_dict.keys())
                            ]

                            log_prob_list.append(log_prob)

                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold_"
                    + str(n_states)
                    + "_states_prediction_all_prob_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(all_proba_dict, f)

                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold_"
                    + str(n_states)
                    + "_states_prediction_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(pred_dict, f)

                init_prob = {}
                for task in tasks:
                    first_state = []
                    for _ in range(n_states):
                        first_state.append(0)
                    trans_matrix = np.zeros((n_states, n_states))
                    for subj, shifted_dict in pred_dict[task].items():
                        for shift_time, feature in shifted_dict.items():
                            # print(feature)
                            for idx, state in enumerate(feature[:-1]):
                                if idx == 0:
                                    first_state[state] += 1
                                # if state != feature[idx + 1]:
                                trans_matrix[state][feature[idx + 1]] += 1
                    first_state = np.array(first_state) / np.array(first_state).sum()
                    init_prob[task] = first_state
                    row_sum = trans_matrix.sum(axis=1)
                    normalized_trans_matrix = trans_matrix / row_sum[:, np.newaxis]
                    # print(np.round(normalized_trans_matrix,3))
                    # print(np.sum(normalized_trans_matrix, axis = 1))

                    # features = ['front leg', 'back leg', 'front hand', 'back hand']
                    df = pd.DataFrame(
                        data=np.vstack(
                            (np.round(normalized_trans_matrix, 3), first_state)
                        )
                    )
                    # df = df.rename(columns=state_name_dict)
                    df.columns = list(state_name_dict.values())
                    index = list(state_name_dict.values())
                    # index[n_states] = "init_prob"
                    index.append("init_prob")
                    df.index = index  # = df.rename(index=index)
                    file_name = (
                        "transmat_" + str(task) + "_state_" + str(n_states) + ".csv"
                    )
                    save_csv(df, file_path, file_name)

                subj_list = list(task_to_storing_dict["MPS"].keys())
                shift_time_list = np.arange(0, interval_length, shift)

                merged_pred_dict_all = {}
                merged_proba_dict_all = {}
                time_subj_dict_all = {}
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
                        all_proba,
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
                    merged_proba_dict_all[task] = merged_proba_dict_all_task_specific
                    time_subj_dict_all[task] = time_subj_dict_all_task_specific

                toy_pred_list = {}
                for task in tasks:
                    toy_pred_list[task] = {}
                    for subj in subj_list:
                        subj_df = pd.DataFrame()
                        pred = []
                        onset = []
                        offset = []

                        onset.append(
                            time_subj_dict_all[task][subj][0]
                            - shift_time_list[1] * 60000
                        )
                        onset.extend(time_subj_dict_all[task][subj][:-1])
                        offset.extend(time_subj_dict_all[task][subj])
                        pred.extend(merged_pred_dict_all[task][subj])

                        for df_ in task_to_storing_dict[task][subj]:
                            subj_df = pd.concat([subj_df, df_])
                        pred_df = pd.DataFrame(
                            data={"onset": onset, "offset": offset, "pred": pred}
                        )

                        pred_df = merge_toy_pred(pred_df, subj_df)
                        toy_pred_list[task][subj] = pred_df

                merged_pred_w_locomotion = {}
                for task in tasks:
                    merged_pred_w_locomotion[task] = {}
                    pred_by_task_dict = merged_pred_dict_all[task]
                    time_by_task_dict = time_subj_dict_all[task]
                    babymovement_by_task_dict = babymovement_dict[task]
                    for subj, pred in pred_by_task_dict.items():
                        # print(subj, task)

                        onset = []
                        onset.append(
                            time_subj_dict_all[task][subj][0]
                            - shift_time_list[1] * 60000
                        )
                        onset.extend(time_subj_dict_all[task][subj][:-1])
                        # if task == "MPS" and 4 in pred_by_task_dict[subj]:
                        # print(subj)

                        df = pd.DataFrame(
                            data={
                                "onset": onset,
                                "offset": time_by_task_dict[subj],
                                "pred": pred_by_task_dict[subj],
                            }
                        )
                        # if subj == 39 and task == 'NMM':
                        #     print(df)
                        merged_pred_w_locomotion[task][subj] = merge_movement(
                            df, babymovement_by_task_dict[subj]
                        )
                    # print(merged_pred_w_locomotion)
                    with open(
                        "./data/interim/20210926"
                        + feature_set
                        + "_"
                        + str(no_ops_time)
                        + "_no_ops_threshold"
                        + str(n_states)
                        + "_states_merged_locomotion_"
                        + str(interval_length)
                        + "_min.pickle",
                        "wb+",
                    ) as f:
                        pickle.dump(merged_pred_w_locomotion, f)
                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold"
                    + str(n_states)
                    + "_states_merged_prediction_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(merged_pred_dict_all, f)

                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold"
                    + str(n_states)
                    + "_states_merged_prediction_prob_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(merged_proba_dict_all, f)

                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold"
                    + str(n_states)
                    + "_states_time_arr_dict_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(time_subj_dict_all, f)

                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_theshold_"
                    + str(n_states)
                    + "_states_toy_pred_dict_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(toy_pred_list, f)

                with open(
                    "./data/interim/20210926"
                    + feature_set
                    + "_"
                    + str(no_ops_time)
                    + "_no_ops_threshold"
                    + str(n_states)
                    + "_states_merged_locomotion_"
                    + str(interval_length)
                    + "_min.pickle",
                    "wb+",
                ) as f:
                    pickle.dump(merged_pred_w_locomotion, f)
