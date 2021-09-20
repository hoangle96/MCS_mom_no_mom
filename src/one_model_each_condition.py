import numpy as np 
import pickle 
from pathlib import Path
from variables import tasks
from hmm_model import convert_to_int, convert_to_list_seqs, save_csv, init_hmm
import pandas as pd
from all_visualization_20210824 import rank_state

shift = .5
with open('./data/interim/20210718_babymovement.pickle', 'rb') as f:
    babymovement_dict = pickle.load(f)

for feature_set in ['n_new_toy_ratio']:#, 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio', 'new_toy_play_time_ratio', 'fav_toy_till_now']:
# for feature_set in ['new_toy_play_time_ratio']:
    print(feature_set)
    for no_ops_time in [10, 5, 7]:
        print('no_ops_time', no_ops_time)
        for interval_length in [1.5, 2, 1]:
            print('interval_length', interval_length)

            with open("./data/interim/20210907_"+str(no_ops_time)+"_no_ops_threshold_feature_dict_with_"+feature_set+"_"+str(interval_length)+"_min.pickle", 'rb') as f:
                feature_dict = pickle.load(f)

            with open("./data/interim/20210907_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
                time_arr_dict = pickle.load(f)

            with open("./data/interim/20210907_"+str(no_ops_time)+"_no_ops_threshold_label_"+str(interval_length)+"_min.pickle", 'rb') as f:
                labels_dict = pickle.load(f)

            with open('./data/interim/20210824_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
                task_to_storing_dict = pickle.load(f)

            for task in tasks:
                print(task)
                n_features = 4
                shift_time_list = np.arange(0, interval_length, shift)

                len_list = []

                input_list = np.empty((0, n_features))

                for subj, shifted_df_dict in feature_dict[task].items():
                    # for shift_time in shift_time_list:
                    for shift_time, feature_vector in shifted_df_dict.items():
                        feature_vector = shifted_df_dict[shift_time]
                        input_list = np.vstack((input_list, feature_vector))
                        len_list.append(len(feature_vector))

                all_labels = []
                for subj, shifted_sequence in labels_dict[task].items():
                    for shift_time, label in shifted_sequence.items(): 
                        all_labels.append(label)

                toy_switch_bins = [0, 5, 10, 15]
                n_bin_ep_rate = range(len(toy_switch_bins))
                discretized_toy_switch_rate = np.digitize(input_list[:,0], toy_switch_bins, right = False)
                discretized_n_toys = np.where(input_list[:,1] > 4, 4, input_list[:,1])

                if feature_set == 'n_new_toy_ratio' or feature_set == 'n_new_toy_ratio_and_fav_toy_till_now' or feature_set == 'new_toy_play_time_ratio':
                    new_toys_bin = [0, .2, .4, .6, .8]
                    discretized_n_new_toys = np.digitize(input_list[:,2].copy(), new_toys_bin, right = False)
                elif feature_set == 'fav_toy_till_now':
                    discretized_n_new_toys = np.where(input_list[:,2] > 4, 4, input_list[:,2])
                
                fav_toy_bin = [0, .2, .4, .6, .8]

                fav_toy_rate_discretized = np.digitize(input_list[:,3].copy(), fav_toy_bin, right = False)

                discretized_input_list = np.hstack((discretized_toy_switch_rate.reshape((-1,1)),\
                                                    discretized_n_toys.reshape((-1,1)),\
                                                    discretized_n_new_toys.reshape((-1,1)),\
                                                    fav_toy_rate_discretized.reshape((-1,1))))

                with open("./data/interim/20210907_"+feature_set+'_'+str(no_ops_time)+"_no_ops_threshold_discretized_input_list_"+str(interval_length)+"_min_"+task+".pickle", 'wb+') as f:
                    pickle.dump(discretized_input_list, f)
                list_seq = convert_to_list_seqs(discretized_input_list, len_list)
                list_seq = convert_to_int(list_seq)

                seed = 1
                for n_states in [5]:#range(4, 7):
                    print('n_states', n_states)
                    model = init_hmm(n_states, discretized_input_list.T, seed)
                    model.bake()

                    # freeze the no_toys distribution so that its parameters are not updated. 
                    # "no_toys" state params are set so that all of the lowest bins = 0
                    for s in model.states:
                        if s.name == "no_toys":
                            for p in s.distribution.parameters[0]:
                                p.frozen = True
                    model.fit(list_seq, labels = all_labels)

                    model_file_name = "model_20210907_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states'+str(task)+'.pickle'
                    model_file_path = Path('./models/hmm/20210907/'+feature_set)/model_file_name
                    Path('./models/hmm/20210907/'+feature_set).mkdir(parents = True, exist_ok = True)
                    with open(model_file_path, 'wb+') as f:
                        pickle.dump(model, f)

                    state_name_dict = rank_state(model)
                    
                    data = []

                    index_list = [[],[]]
                    
                    features_obs_dict = {0: len(toy_switch_bins) , 1: 5, 2: len(new_toys_bin), 3: 5}

                    for i in range(n_features):
                        single_list = np.empty((features_obs_dict[i], n_states))
                        for state_idx, state_i in enumerate(range(n_states)):
                            observation_dict = model.states[state_i].distribution.parameters[0][i].parameters[0]
                            for idx,k in enumerate(observation_dict.keys()):
                                single_list[idx, state_idx] = np.round(observation_dict[k], 2)
                        index_list[0].extend([i]*len(observation_dict.keys()))
                        index_list[1].extend([i for i in observation_dict.keys()])

                        data.extend(single_list)

                    tuples = list(zip(*index_list))
                    index = pd.MultiIndex.from_tuples(tuples, names=['feature', 'observation'])
                    df = pd.DataFrame(data, index = index, columns = ['state '+str(state_name_dict[i]) for i in range(n_states)])
                    file_path = Path('/scratch/mom_no_mom/reports/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/state_'+str(n_states))
                    file_path.mkdir(parents = True, exist_ok = True)
                    file_name = 'mean_'+str(n_states)+"_states"+str(task)+".csv"
                    save_csv(df, file_path, file_name)

                    # save the transition matrix for all
                    trans_matrix = pd.DataFrame(np.round(model.dense_transition_matrix()[:n_states+1,:n_states],2))
                    file_name = 'trans_matrix_'+str(n_states)+"_states_seed_"+str(seed)+'_'+str(interval_length)+'_min'+str(task)+'.csv'
                    trans_matrix = trans_matrix.rename(state_name_dict, axis=1) 
                    index = state_name_dict
                    index[n_states] = 'init_prob'
                    trans_matrix = trans_matrix.rename(index, axis = 0) 
                    save_csv(trans_matrix, file_path, file_name)

                    i = 0
                    input_dict = {}

                    for subj, shifted_df_dict in feature_dict[task].items():
                        if subj not in input_dict.keys():
                            input_dict[subj] = {}

                        # for shift_time, feature_vector in shifted_df_dict.items():
                        for shift_time in shift_time_list:
                            input_dict[subj][shift_time] = list_seq[i]
                            i += 1

                    total_log_prob = 0
                    log_prob_list = []
                    pred_dict = {}
                    proba_dict = {}
                    all_proba_dict = {}

                    for subj, shifted_dict in input_dict.items():
                        if subj not in pred_dict.keys():
                            pred_dict[subj] = {}
                            proba_dict[subj] = {}
                            all_proba_dict[subj] = {}

                        for shift_time, feature_vector in shifted_dict.items():
                            label= model.predict(feature_vector)
                            pred_dict[subj][shift_time] = label

                            # if 4 in label:
                                # print(feature_vector, label)
                            proba_dict[subj][shift_time] = np.amax(model.predict_proba(feature_vector), axis = 1)
                            log_prob = model.log_probability(feature_vector)
                            all_proba_dict[subj][shift_time] = model.predict_proba(feature_vector)
                            
                            log_prob_list.append(log_prob)

                    with open('./data/interim/20210907_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min'+str(task)+'.pickle', 'wb+') as f:
                        pickle.dump(all_proba_dict, f)

                    with open('./data/interim/20210907_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min'+str(task)+'.pickle', 'wb+') as f:
                        pickle.dump(pred_dict, f)