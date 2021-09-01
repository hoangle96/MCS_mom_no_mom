import numpy as np 
import pandas as pd
from variables import tasks, state_color_dict
import pickle 

from visualization import draw_timeline_with_prob_to_compare
import os
from pathlib import Path 
from all_visualization import rank_state

if __name__ == '__main__':
    shift = .25

    for feature_set in ['n_new_toy_ratio', 'fav_toy_till_now', 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio']:
        for no_ops_time in [10, 7]:
            with open('./data/interim/20210805_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
                task_to_storing_dict = pickle.load(f)

            print('no_ops_time', no_ops_time)
            for interval_length in [1.5]:
                print('interval_length', interval_length)
                shift_time_list = np.arange(0,interval_length, shift)
                
                with open("./data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_feature_dict_with_"+feature_set+"_"+str(interval_length)+"_min.pickle", 'rb') as f:
                    feature_dict = pickle.load(f)

                with open("./data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
                    time_arr_dict = pickle.load(f)

                with open("./data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_label_"+str(interval_length)+"_min.pickle", 'rb') as f:
                    labels_dict = pickle.load(f)

                # with open("./data/interim/20210815_"+feature_set+'_'+str(no_ops_time)+"_no_ops_threshold_discretized_input_list_"+str(interval_length)+"_min.pickle", 'rb') as f:
                #     discretized_input_list = pickle.load(f)
                
                for n_states in range(5, 7):
                    print("n_states", n_states)
                    with open('./data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        pred_dict = pickle.load(f)

                    with open('./data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        all_proba_dict = pickle.load(f)

                    with open('./data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        merged_pred_dict_all = pickle.load(f)

                    with open('./data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        merged_proba_dict_all = pickle.load(f)

                    with open('./data/interim/20210815_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        time_subj_dict_all = pickle.load(f)
                    
                  

                    model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
                    model_file_path = Path('./models/hmm/20210815/'+feature_set)/model_file_name
                    with open(model_file_path, 'rb') as f:
                        model = pickle.load(f)
                    state_name_dict = rank_state(model)

                    subj_list = list(task_to_storing_dict['MPS'].keys())


                    toy_colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                                    'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                                    'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'violet', 'bucket': 'navy', 'no_toy': "slategrey"}

                    for subj in subj_list:
                        for task in tasks:
                            df = pd.DataFrame()
                            for df_ in task_to_storing_dict[task][subj]:
                                df = pd.concat([df, df_])
                            
                            list_of_state_list = []
                            list_of_time_list = []
                            list_of_prob_list = []
                            for shift_time in np.arange(0, interval_length, shift):
                                if shift_time in [0.0, 1.0, 2.0]:
                                    shift_time = int(shift_time) 
                                path = Path('./figures/hmm/state_distribution_20210815/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states/all_sequences/'+task+'/')
                                path.mkdir(parents=True, exist_ok=True)

                                
                                pred_state_list = pred_dict[task][subj][shift_time]
                                state_name_list = [state_name_dict[s] for s in pred_state_list]

                                list_of_state_list.append(state_name_list)
                                # print(list_of_state_list)
                                time_list = time_arr_dict[task][subj][shift_time]
                                list_of_time_list.append(time_list)
                                prob_list = all_proba_dict[task][subj][shift_time]
                                list_of_prob_list.append(prob_list)
                            if len(time_list) < 2:
                                print(subj, task, shift_time)
                            if len(time_list) != len(prob_list):
                                print(subj, task, shift_time)
                            fig_name = './figures/hmm/state_distribution_20210815/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states/all_sequences/'+task+'/'+str(subj)+".png"
                            # draw_timeline_with_merged_states(subj, df, pred_state_list, time_list, state_name_dict[no_ops_time][interval_length][n_states], fig_name= fig_name, gap_size = interval_length, show=False)
                            draw_timeline_with_prob_to_compare(title = str(subj) + " window size: " + str(interval_length) + " no ops threshold "+ str(no_ops_time) + " shift time: " + str(shift*60)+"s", \
                                                        df = df, list_of_state_list = list_of_state_list, list_of_time_list = list_of_time_list,\
                                                        state_name = state_name_dict, fig_name= fig_name, gap_size = interval_length,\
                                                        state_color_dict= state_color_dict, list_of_prob_list = list_of_prob_list, shift = shift)
