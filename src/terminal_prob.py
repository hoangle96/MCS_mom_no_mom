import numpy as np
import pickle 
# import 
from pathlib import Path 
from all_visualization_20210824 import rank_state
from variables import tasks

if __name__ == '__main__':
    feature_set = 'n_new_toy_ratio'
    interval_length = 1.5
    n_states = 5
    no_ops_time = 10
    model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
    model_file_path = Path('./models/hmm/20210824__30s_offset/'+feature_set)/model_file_name
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    state_name_dict = rank_state(model)

    with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
        all_proba_dict =pickle.load(f)

    with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
        pred_dict = pickle.load(f)

    print(state_name_dict)
    for task in tasks:
        last_state = []
        for _ in range(n_states):
            last_state.append(0)
        for subj, shifted_dict in pred_dict[task].items():    
            for shift_time, feature in shifted_dict.items():
                last_state_ = feature[-1]
                last_state[last_state_] += 1
        last_state = np.array(last_state)/np.array(last_state).sum()
        # last_state = 
        arranged = {v: last_state[k] for k, v in state_name_dict.items()}
        print(task, arranged)
            
        
