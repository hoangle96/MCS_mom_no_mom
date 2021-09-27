import numpy as np 
import pandas as pd
from variables import tasks, condition_name, state_color_dict, stationary_toys_list, mobile_toys_list, state_color_dict_shades
import pickle 
from merge import merge_segment_with_state_calculation_all
from visualization import draw_toy_state, draw_distribution, draw_timeline_with_merged_states, draw_state_distribution, draw_toy_state_with_std, draw_infant_each_min_matplotlib, draw_mean_state_locotion_across_conditions, draw_timeline_with_prob_to_check, draw_mean_state_locotion_across_conditions_separate_mean_std
import os
from pathlib import Path 
from all_visualization_20210824 import rank_state

if __name__ == '__main__':
    shift = .5
    mobile_toys_list.append('no_toy')
    stationary_toys_list.append('no_toy')

    for feature_set in ['n_new_toy_ratio']:#, 'fav_toy_till_now', 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio']:
    # for feature_set in ['new_toy_play_time_ratio']:
        for no_ops_time in [10,7, 5]:
            with open('./data/interim/20210824_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
                task_to_storing_dict = pickle.load(f)

            print('no_ops_time', no_ops_time)
            for interval_length in [1.5, 2, 1]:
                # if (no_ops_time == 5 and interval_length > 1) or no_ops_time != 5:
                    # print('interval_length', interval_length)
                shift_time_list = np.arange(0,interval_length, shift)

                
                with open("./data/interim/20210824_"+str(no_ops_time)+"_no_ops_threshold_feature_dict_with_"+feature_set+"_"+str(interval_length)+"_min.pickle", 'rb') as f:
                    feature_dict = pickle.load(f)

                with open("./data/interim/20210824_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
                    time_arr_dict = pickle.load(f)

                with open("./data/interim/20210824_"+str(no_ops_time)+"_no_ops_threshold_label_"+str(interval_length)+"_min.pickle", 'rb') as f:
                    labels_dict = pickle.load(f)

                for task in tasks:
                    with open("./data/interim/20210907_"+feature_set+'_'+str(no_ops_time)+"_no_ops_threshold_discretized_input_list_"+str(interval_length)+"_min_"+task+".pickle", 'rb') as f:
                        discretized_input_list = pickle.load(f)
                
                    for n_states in [5]:#range(4, 7):
                        print('states', n_states)
                        with open('./data/interim/20210904_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min'+task+'.pickle', 'rb') as f:
                            pred_dict = pickle.load(f)

                        with open('./data/interim/20210904_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min'+task+'.pickle', 'rb') as f:
                            all_proba_dict = pickle.load(f)

                        model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states'+str(task)+'.pickle'
                        model_file_path = Path('./models/hmm/20210907/'+feature_set)/model_file_name
                        with open(model_file_path, 'rb') as f:
                            model = pickle.load(f)
                        state_name_dict = rank_state(model)

                        # subj_list = list(task_to_storing_dict['MPS'].keys())


                        toy_colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                                        'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                                        'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'violet', 'bucket': 'navy', 'no_toy': "slategrey"}

                        ### state pred figures

                        if feature_set == 'n_new_toy_ratio' or feature_set == 'n_new_toy_ratio_and_fav_toy_till_now':
                            x_ticks_dict = {0: ["[0, 4)", '[4, 8)', '[8, 12)', '[12+'], 1: ['0', '1', '2', '3', '4+'], 2: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
                            feature_names = ["# toys switches", "# toys", "# new toys ratio", 'fav toy ratio']
                            feature_values = {0: range(1,5), 1: range(5), 2: range(1, 6), 3: range(1,6)}

                        elif feature_set == 'fav_toy_till_now':
                            x_ticks_dict = {0: ["[0, 4)", '[4, 8)', '[8, 12)', '[12+'], 1: ['0', '1', '2', '3', '4+'], 2: ['0', '1', '2', '3', '4+'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
                            feature_names = ["# toys switches", "# toys", "# new toys", 'fav toy ratio']
                            feature_values = {0: range(1,5), 1: range(5), 2: range(5), 3: range(1,6)}
                        elif feature_set == 'new_toy_play_time_ratio':
                            x_ticks_dict = {0: ["[0, 4)", '[4, 8)', '[8, 12)', '[12+'], 1: ['0', '1', '2', '3', '4+'], 2: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]'], 3: ["[0, .2)", '[.2, .4)', '[.4, .6)', '[.6, .8)', '[.8, 1]']}
                            feature_names = ["# toys switches", "# toys", "new toys play time ratio", 'fav toy ratio']
                            feature_values = {0: range(1,5), 1: range(5), 2: range(1, 6), 3: range(1,6)}

                        n_features = 4
                        flatten_pred = []
                        # flatten_pred_dict = {}
                        Path('./figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/").mkdir(parents=True, exist_ok=True)
                        # for task in tasks:
                            # flatten_pred_dict[task] = []
                            # task_specific_pred_dict = pred_dict[task]
                        for subj, subj_dict in pred_dict.items():
                            # print(subj_dict)
                            for shift_time, pred in subj_dict.items():
                                flatten_pred.extend(pred)
                                    # flatten_pred_dict[task].extend(pred)
                        
                        fig_path = './figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/distribution_time_in_state_indv_model_"+task+".png"
                        draw_state_distribution(flatten_pred, n_states, state_name_dict, condition_name[task], state_color_dict_shades, fig_path)
                        # fig_path = './figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/state_distribution.png"  
                        # draw_state_distribution(flatten_pred, n_states, state_name_dict, "Distribution of time spent in each state, " +'\n'+ str(no_ops_time) + 's threshold,window size ' + str(interval_length) +" min", state_color_dict_shades, fig_path)

                        fig_path = './figures/hmm/20210824/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+"_states/emission_distribution_"+task+".png"
                        draw_distribution(n_features, state_name_dict, discretized_input_list, np.array(flatten_pred), str(no_ops_time) + 's threshold, window size ' + str(interval_length) +" min",feature_names, x_ticks_dict, feature_values, state_color_dict_shades, fig_path)
                            
