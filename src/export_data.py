import numpy as np
import pickle
from all_visualization_20210824 import rank_state
from pathlib import Path
from variables import tasks

interval_length = 1.5
no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
with open('./data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_all_pred_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    all_prob_dict_all = pickle.load(f)
with open('./data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    merged_pred_dict_all = pickle.load(f)

model_file_name = "model_20210907_"+feature_set+"_" + \
    str(interval_length)+"_interval_length_"+str(no_ops_time) + \
    "_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('./models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)
print(state_name_dict)

convert_pred_dict = {}
converted_proba_dict = {}
marginal_distribution = {}

for task in tasks:
    convert_pred_dict[task] = {}
    converted_proba_dict[task] = {}
    for subj, pred in merged_pred_dict_all[task].items():
        convert_pred_dict[task][subj] = np.array(
            [state_name_dict[i] for i in pred]).astype(int)
        # print(pred)
        # print(convert_pred_dict[task][subj])
        change_prob = np.array(all_prob_dict_all[task][subj])[
            :, list(state_name_dict.keys())]
        converted_proba_dict[task][subj] = change_prob

    all_pred_task = np.hstack(list(convert_pred_dict[task].values()))
    unique_state, cnt = np.unique(all_pred_task, return_counts=True)
    cnt = cnt/cnt.sum()
    marginal_distribution[task] = {}
    for state in state_name_dict.values():
        marginal_distribution[task][int(state)] = cnt[unique_state == int(
            state)].item() if int(state) in unique_state else 0

with open("./data/result/pickle_files/pred_10s_1.5min_5states_20210907.pickle", "wb+") as f:
    pickle.dump(convert_pred_dict, f)

with open("./data/result/pickle_files/pred_prob_10s_1.5min_5states_20210907.pickle", "wb+") as f:
    pickle.dump(converted_proba_dict, f)

with open("./data/result/pickle_files/marginal_distribution_10s_1.5min_5states_20210907.pickle", "wb+") as f:
    pickle.dump(marginal_distribution, f)
