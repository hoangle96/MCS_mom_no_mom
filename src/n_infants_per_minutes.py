import pickle
import numpy as np 
from visualization import draw_infant_each_min_matplotlib
from variables import tasks, condition_name
from all_visualization import rank_state
from pathlib import Path 

def get_longest_item(dictionary):
    return max((len(v)) for _,v in dictionary.items())

if __name__ == "__main__":
    for no_ops_time in [5, 7, 10]:
        for interval_length in [1, 1.5, 2]:
            n_states = 5
            with open('./data/interim/20210805_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    merged_pred_dict_all = pickle.load(f)

            model_file_name = "model_20210805_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
            model_file_path = Path('./models/hmm/20210805/')/model_file_name
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
            state_name_dict = rank_state(model)
            
            # print(merged_pred_dict_all["MPS"])
            # print(state_name_dict)
            
            cnt_dict_task_specific = {}
            for task in tasks:
                cnt_dict_task_specific[task] = {}
                len_ = get_longest_item(merged_pred_dict_all[task])
                for i in range(n_states):
                    cnt_dict_task_specific[task][str(i)] = [0]*len_
            
            for task in tasks:
                for subj, state_list in merged_pred_dict_all[task].items():
                    for state_key, state_name in state_name_dict.items():
                        # state_list = np.where(state_list == state_key, state_name, state_list)
                        named_state_list = [state_name_dict[s] for s in state_list]
                    # print(state_list)
                    # print(named_state_list)

                    for idx, state in enumerate(named_state_list):
                        cnt_dict_task_specific[task][state][idx] += 1
                
                focus_state = np.array(cnt_dict_task_specific[task]["1"]) + np.array(cnt_dict_task_specific[task]["2"]) 
                explore_state = np.array(cnt_dict_task_specific[task]["3"]) + np.array(cnt_dict_task_specific[task]["4"]) 
                file_name = './figures/hmm/state_distribution_20210805/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/'+"n_infants_each_state_per_min_"+task+'.png'
                draw_infant_each_min_matplotlib(focus_state, explore_state, cnt_dict_task_specific[task]["0"], condition_name[task], file_name)