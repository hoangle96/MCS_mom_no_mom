import numpy as np 
import pandas as pd
from variables import tasks
import pickle 
from visualization import draw_toy_state, draw_distribution, draw_timeline_with_merged_states
from all_visualization_20210824 import rank_state
from pathlib import Path
shift = .25
for feature_set in ['n_new_toy_ratio']:#, 'fav_toy_till_now', 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio', 'new_toy_play_time_ratio']:
    for no_ops_time in [5, 7, 10]:
        with open('./data/interim/20210824_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
            task_to_storing_dict = pickle.load(f)
        subj_list = list(task_to_storing_dict['MPS'].keys())
        print('no_ops_time', no_ops_time)
        for interval_length in [1, 1.5, 2]:
            print('interval_length', interval_length)
            shift_time_list = np.arange(0, interval_length, shift)


            with open("./data/interim/20210824_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min.pickle", 'rb') as f:
                feature_dict = pickle.load(f)

            with open("./data/interim/20210824_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
                time_arr_dict = pickle.load(f)

            with open("./data/interim/20210824_"+str(no_ops_time)+"_no_ops_threshold_label_"+str(interval_length)+"_min.pickle", 'rb') as f:
                labels_dict = pickle.load(f)

            # with open("./data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_discretized_input_list_"+str(interval_length)+"_min.pickle", 'rb') as f:
            #     discretized_input_list = pickle.load(f)
            
            for n_states in range(4, 7):
                with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    pred_dict = pickle.load(f)

                with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    merged_pred_dict_all = pickle.load(f)

                with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    merged_proba_dict_all = pickle.load(f)

                with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    time_subj_dict_all = pickle.load(f)
                
                with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_theshold_'+str(n_states)+'_states_toy_pred_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    toy_pred_list = pickle.load(f)

                model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
                model_file_path = Path('./models/hmm/20210824__30s_offset/'+feature_set+'/')/model_file_name
                with open(model_file_path, 'rb') as f:
                    model = pickle.load(f)
                state_name_dict = rank_state(model)

                Path('./data/result/prediction/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/state_'+str(n_states)).mkdir(parents = True, exist_ok = True)
                for subj in subj_list:
                    pred = []
                    onset = []
                    offset = []
                    for task in tasks:
                        onset.append(time_subj_dict_all[task][subj][0] - shift_time_list[1]*60000)
                        onset.append(time_subj_dict_all[task][subj][0])

                        onset.extend([s if s - time_subj_dict_all[task][subj][idx -1] <=shift_time_list[1]*60000 else s - shift_time_list[1]*60000 for idx, s in enumerate(time_subj_dict_all[task][subj][1:-1], 1)])
                        offset.extend(time_subj_dict_all[task][subj])
                        pred.extend(merged_pred_dict_all[task][subj])
                    
                    # print(np.diff(onset))
                    # print(np.diff(offset))
                    # print(len(onset) == len(offset))
                    df = pd.DataFrame({"onset":onset, "offset": offset, "pred": pred})
                    df['pred'] = df['pred'].replace(state_name_dict)
                    
                    df = df.sort_values(by=['onset'])
                    df.to_csv('./data/result/prediction/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/state_'+str(n_states)+'/'+str(subj)+'.csv', index = False)