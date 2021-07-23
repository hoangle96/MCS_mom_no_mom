import numpy as np
from sklearn.model_selection import KFold
import pickle
import os 
import pomegranate as pom
import sys
from sklearn.utils import check_random_state
from variables import toys_dict, tasks, toys_list


def discritize_with_sub(feature, threshold = 4):
    """
    Cap out the values at the higher end. Used for 'n_toys' and 'n_new_toys' features
    """
    return np.where(feature > threshold, threshold, feature)

def discritize_with_bins(feature, bins_):
    """
    Group values into different bins. Used for 'n_toy_switches" and 'fav_toy_ratio'
    """
    return np.digitize(feature, bins_, right = False)

def discritize_toy_iou(feature, discretizer):
    return discretizer.transform(feature).reshape((-1,))

def convert_to_int(list_to_convert):
    converted_list = []
    for i in list_to_convert:
        i = i.astype(int)
        converted_list.append(i)
    return converted_list

def convert_to_list_seqs(big_seq, len_array):
    big_seq_to_slice = big_seq.copy()
    list_of_seqs = []
    
    for k in len_array:
        list_of_seqs.append(big_seq_to_slice[:k])
        big_seq_to_slice = big_seq_to_slice[k:]
    return list_of_seqs

def save_csv(df, file_path, file_name):
    if not os.path.exists(file_path):
        file_path.mkdir(parents=True)
    save_path = file_path / file_name
    df.to_csv(save_path)

def create_independent_dist(feature, seed):
    unique_val = np.unique(feature)
    init_dict = {}
    random_state = check_random_state(seed)
    init_prob = random_state.rand(len(unique_val),1)
    init_prob = init_prob/init_prob.sum()
    
    for idx, i in enumerate(unique_val):
        init_dict[int(i)] = init_prob[idx].item()
    return pom.DiscreteDistribution(init_dict)

def create_no_ops_state(feature):
    unique_val = np.unique(feature)
    init_dict = {}
    # print(unique_val)
    for idx, i in enumerate(unique_val):
        if idx == 0:
            init_dict[int(i)] = 1
        else:
            init_dict[int(i)] = 0
    return pom.DiscreteDistribution(init_dict)


def create_dist_for_states(n_states, feature_list, seed):
    distributions = []
    i = 0
    for s in range(n_states):
        if s == 0:
            dist_list = []
            for f in feature_list:
                dist = create_no_ops_state(f)
                dist_list.append(dist)
            distributions.append(pom.IndependentComponentsDistribution(dist_list))

        else:
            dist_list = []
            for f in feature_list:
                dist = create_independent_dist(f, i)
                i += 1
                dist_list.append(dist)
            distributions.append(pom.IndependentComponentsDistribution(dist_list))
    return distributions

def init_hmm(n_components, feature_list, seed):
    random_state_trans = check_random_state(seed**seed)
    transitions = random_state_trans.rand(n_components, n_components)
    transitions = transitions/transitions.sum()
    
    random_state_start = check_random_state(seed**2)
    starts = random_state_start.rand(n_components)
    starts = starts/starts.sum()
    distributions = create_dist_for_states(n_components, feature_list, seed)
    state_names = ["no_toys"] + [None]*(n_components-1)
    model = pom.HiddenMarkovModel.from_matrix(transitions, distributions, starts, state_names = state_names)   
    return model

def kfold_each_window_size(list_of_feature, all_labels, n_bin_toy_switch, max_n_states):
    score_dict = {}
    
    for n_states in range(4, max_n_states):
        print(str(n_states) + " states")
        
        score_dict[n_states] = []
        kf = KFold(n_splits = 3)
        for train_index, test_index in kf.split(list_of_feature):
            # print(train_index.tolist(), len(test_index))
            train_trials = np.array(list_of_feature)[train_index]
            test_trials = np.array(list_of_feature)[test_index]
            train_labels = np.array(all_labels)[train_index].tolist()
            train_labels = [t.tolist() for t in train_labels]

            len_train = [len(i) for i in train_trials]
            train_input = np.concatenate(train_trials)


            train_input[:,0] = discritize_with_bins(train_input[:,0], bins_ = n_bin_toy_switch) 
            train_input[:,1] = discritize_with_sub(train_input[:,1]) 
            train_input[:,2] = discritize_with_sub(train_input[:,2]) 
            train_input[:,3] = discritize_with_bins(train_input[:,3], bins_ =  [0, .2, .4, .6, .8]) 

            train_seq = convert_to_list_seqs(train_input, len_train)
            train_seq = convert_to_int(train_seq)

            # train_labels = convert_to_list_seqs(train_labels, len_train)
            
            len_test = [len(i) for i in test_trials]
            test_input = np.concatenate(test_trials)

            test_input[:,0] = discritize_with_bins(test_input[:,0], bins_ = [0,5,10,15]) 
            test_input[:,1] = discritize_with_sub(test_input[:,1]) 
            test_input[:,2] = discritize_with_sub(test_input[:,2]) 
            test_input[:,3] = discritize_with_bins(test_input[:,3], bins_ =  [0, .2, .4, .6, .8]) 
            
            test_seq = convert_to_list_seqs(test_input, len_test)
            test_seq = convert_to_int(test_seq)

            
            model = init_hmm(n_states, train_input.T, seed = 1)
            for s in model.states:
                if s.name == "no_toys":
                    for p in s.distribution.parameters[0]:
                        p.frozen = True
            model.bake()
            model.fit(train_seq, labels = train_labels)

            score_ = []
            for seq in test_seq:
                log_prob = model.log_probability(seq)
                if log_prob != -np.inf:
                    score_.append(log_prob)
            score_dict[n_states].append(score_)
    return score_dict