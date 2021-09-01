import numpy as np, pickle
from variables import toys_dict, tasks, toys_list
import pandas as pd
from visualization import draw_timeline_with_merged_states, draw_plain_timeline, draw_plain_timeline_with_feature_discretization
import itertools

def merge_toy_pred(pred_df, subj_df):
    """
    Merging toys-timestamp and final prediction timestamp. 
    The idea is to create a place holder every time there is a change in the toyset or change in the prediction. 
    Parameters:
        pred_df: Pandas dataframe containing the prediction. Format: pd.DataFrame({onset, offset, pred})
        subj_df: Pandas dataframe containing the toy timestamp. Format: pd.DataFrame({onset, offset, pred})
    Return:
        Merged dataframe containing both info

    """
    # Get all timestamp and sort them
    all_onsets = list(set(pred_df['onset'].unique().tolist() + subj_df['onset'].unique().tolist()))
    all_offsets = list(set(pred_df['offset'].unique().tolist() + subj_df['offset'].unique().tolist()))
    time = list(set(all_onsets + all_offsets))
    time.sort()

    toys_list = []
    pred_list = []
    onset_list = []
    offset_list = []

    for idx, onset in enumerate(time):
        if idx != len(time) - 1:
            offset = time[idx+1] 
            onset_list.append(onset)
            offset_list.append(offset)

            pred = pred_df.loc[(pred_df.loc[:,'onset'] <= onset) & (pred_df.loc[:,'offset'] >= offset), 'pred'].tolist()
            pred_list.append(pred[0])
            toys = subj_df.loc[(subj_df.loc[:,'onset'] <= onset) & (subj_df.loc[:,'offset'] >= offset), 'toy'].tolist()
            if 'no_ops' in toys:
                toys = [t for t in toys if 'no_ops' not in t]
            toys_list.append(list(set(itertools.chain.from_iterable(toys))))

    return pd.DataFrame({'onset': onset_list, 'offset': offset_list, 'pred': pred_list, 'toys':toys_list })

def calculate_states_all(prob_list: np.ndarray) -> np.ndarray:
    """
    Normalize the probability of all the state at the current time stamp.
    1. Take the sum of probability for each state across all sequences.
    2. Normalize

    Parameter:
        a: Probability matrix for all states from all sequences. Each column is a state. Each row is a sequence .Shape: (n x n_states)
    Return: 
        Normalized probability distribution
    
    """
    a = np.sum(prob_list, axis = 0)
    # print(a/a.su)
    return a/a.sum()
    
def get_indv_state_argmax_all(pred_indv_dict, proba_indv_dict, time_indv_dict, shift_array, begin_time, end_time, window_size, n_states, shift_interval):
    """
    Find the final state prediction from different offseted sequence.
    Have a sliding pointer, starting at 'begin_time', incrementing by 'shift_interval', ending at 'end_time'
    At each time, get all the probability of the states of the interval that ends/is at the timestamp from all different offseted sequences
    then normalized by calculate_states_all
    Parameters:
        pred_indv_dict: prediction of states. Nested dictionary, {offset: prediction}: list of int
        proba_indv_dict: probability of each state. Nested dictionary, {offset: probability}, probability: list of probability for all states
        time_indv_dict: time dictionary of  
        shift_array: list of all offset time
        
        begin_time: begin of the trial/session (in ms)
        end_time: end of the trial/session (in ms)
        n_states: number of states in the hmm model
        shift_interval:  (int)
        window_size: window size, in minute
        n_states: (int)
        shift_interval: (int) diffeenece in offset
    Returns:
        merged_pred_dict: final prediction {subject: {offset: {prediction}}, prediction: list of int
        merged_proba_dict: final probability {subject: {offset: {probability}}, probability: list of int
        time_subj_dict: timestamp where each state in 'prediction' ends
    """
    time_list = []
    merged_indv_pred_list = []
    merged_indv_proba_list = []
    all_prob = []
    ptr = begin_time + shift_interval
    # print(begin_time, end_time)
    idx_shift = {}
    
    while  ptr - end_time < window_size*1/2*60000:

        if ptr > end_time:
            ptr = end_time
        prob_list = []
        pred_list = []*n_states
        for shift in shift_array:
            begin = time_indv_dict[shift][0] - window_size*60000 
            end = time_indv_dict[shift][-1]
            # print(ptr, begin, end)

            # check for beginning of each shift sequence
            if ptr > begin and ptr <= end:
                # print(ptr, time_indv_dict[shift][-1])
                
                idx = next(idx for idx, value in enumerate(time_indv_dict[shift]) if value >= ptr)
                pred_ = pred_indv_dict[shift][idx]
                idx_shift[shift] = idx
                
                prob_list.append(proba_indv_dict[shift][idx])
        # print(prob_list)
        result_prob = calculate_states_all(np.array(prob_list).reshape((-1, n_states)))
        highest_idx = np.argmax(result_prob)
        all_prob.append(result_prob)
        # if not isinstance(result_prob, list):
        #     result_prob = np.array(result_prob).reshape((1,-1))
        #     print(result_prob, highest_idx)


        merged_indv_pred_list.append(highest_idx)

        merged_indv_proba_list.append(result_prob[highest_idx])
        time_list.append(ptr)
        if ptr == end_time:
            break
        ptr += shift_interval
    return merged_indv_pred_list, merged_indv_proba_list, all_prob, time_list, idx_shift

def smooth_state_presentation(state_list, prob_list):
    for idx, (state, prob) in enumerate(zip(state_list, prob_list)):
        prob = np.round(prob, 2)
        max_prob = np.amax(prob)
        prob_, cnt = np.unique(prob, return_counts = True)
        
        if cnt[prob_==max_prob] > 1:
            max_state = np.where(prob == max_prob)[0]
            for s in max_state:
                if idx > 0 and idx < len(state_list) - 1:
                    if s == state_list[idx-1] or s == state_list[idx+1]: 
                        state_list[idx] = s
                elif idx == 0 and len(state_list) > 1:
                    if s == state_list[1]: 
                        state_list[idx] = s
                elif idx == len(state_list) - 1:
                    if s == state_list[len(state_list) - 2]: 
                        state_list[idx] = s
    return state_list


def merge_segment_with_state_calculation_all(subj_list, shift_array, df_dict, time_arr_dict, pred_dict, prob_dict, window_size, n_states, shift_interval):
    """
    For each subject, find the final state prediction from different offseted sequence
    Parameters:
        subj_list: list of all subjects
        shift_array: list of all offset time
        df_dict: dictionary of the toy-timestamp for each infant. Use this to get the beginning and the end of a session
        time_arr_dict: time dictionary of  
        pred_dict: prediction of states. Nested dictionary, {subject: {offset: prediction}: list of int
        prob_dict: probability of each state. Nested dictionary, {subject: {offset: probability}, probability: list of probability for all states
        window_size: (int)
        n_states: (int)
        shift_interval: difference in the offset
    Returns:
        merged_pred_dict: final prediction {subject: {offset: {prediction}}, prediction: list of int
        merged_proba_dict: final probability {subject: {offset: {probability}}, probability: list of int
        time_subj_dict: timestamp where each state in 'prediction' ends
    """
    merged_pred_dict = {}
    merged_proba_dict = {}
    time_subj_dict = {}
    all_prob_dict = {}
    # global task

    for k in subj_list:
        # print(k)
        if k == 7:
            print('here')
        # if k == 1 and task == 'NMS':
        #     print('here')
        # df = pd.DataFrame()
        idx_dict = {}
        for shift in shift_array:
            idx_dict[shift] = 0
        # print(idx_dict[shift])
        time_list = []
        merged_indv_proba_list = []
        merged_indv_pred_list = []
        all_prob = []
        for df_ in df_dict[k]:
            # df = pd.concat([df, df_])
        
            begin_time = df_.iloc[0,:]
            begin_time = begin_time.loc['onset']
            end_time = df_.iloc[-1,:]
            end_time = end_time.loc['offset']
            
            time_indv_dict = {}
            pred_indv_dict = {}
            proba_indv_dict = {}
            
            for shift in shift_array:
                if shift in idx_dict.keys():
                    idx = idx_dict[shift]
                else:
                    idx = 0
                time_indv_dict[shift] = time_arr_dict[k][shift][idx:]
                pred_indv_dict[shift] = pred_dict[k][shift][idx:]
                proba_indv_dict[shift] = prob_dict[k][shift][idx:]
        # print(time_indv_dict)
        
            merged_indv_pred_list_, merged_indv_proba_list_, all_prob_, time_list_, idx_dict = get_indv_state_argmax_all(pred_indv_dict, \
                proba_indv_dict, time_indv_dict, shift_array, begin_time, end_time, window_size, n_states, shift_interval)
            time_list.extend(time_list_)  
            merged_indv_pred_list.extend(merged_indv_pred_list_)  
            merged_indv_proba_list.extend(merged_indv_proba_list_)
            all_prob.extend(all_prob_)
        
        merged_indv_pred_list = smooth_state_presentation(merged_indv_pred_list, all_prob)

        time_subj_dict[k] = time_list

        merged_proba_dict[k] = merged_indv_proba_list
        merged_pred_dict[k] = merged_indv_pred_list
        all_prob_dict[k] = all_prob
    return merged_pred_dict, merged_proba_dict, time_subj_dict, all_prob_dict

if __name__ == "__main__":
    # with open('./data/interim/20210709_5_states_prediction_1.5_min.pickle', 'rb') as f:
    #     pred_dict = pickle.load(f)
    CHECKING = True
    interval_length = 1.5
    n_states = 5
    no_ops_theshold = 7
    gap_size = 60000*.25
    print(gap_size)
    with open("./data/interim/20210726_"+str(no_ops_theshold)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
        time_arr_dict = pickle.load(f)

    with open('./data/interim/20210726_'+str(no_ops_theshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
        task_to_storing_dict = pickle.load(f)
    
    # with open('./data/interim/20210721_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    #     all_proba_dict = pickle.load(f)

    # with open('./data/interim/20210721_'+str(n_state)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
        # pred_dict = pickle.load(f)

    with open('./data/interim/20210726_'+str(no_ops_theshold)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
        all_proba_dict = pickle.load(f)

    with open('./data/interim/20210726_'+str(no_ops_theshold)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
        pred_dict = pickle.load(f)

    # state name dict for no_ops_threshold of 7s, 1.5 interval, 5 states    
    state_name_dict = {4: "No_toys", 2: "F+", 0: "F", 3: "E", 1:"E+"}

    subj_list = list(task_to_storing_dict['MPS'].keys())
    shift_time_list = np.arange(0, interval_length, 0.25)

    merged_pred_dict_all = {}
    merged_proba_dict_all = {}
    time_subj_dict_all = {}

    for task in tasks:
        print(task)
        merged_df_dict = task_to_storing_dict[task]
        time_arr_shift_dict = time_arr_dict[task]
        pred_subj_dict = pred_dict[task]
        prob_subj_dict = all_proba_dict[task]

        merged_pred_dict_all_task_specific, merged_proba_dict_all_task_specific, time_subj_dict_all_task_specific = merge_segment_with_state_calculation_all(subj_list, shift_time_list, merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size = interval_length, n_states = 5, shift_interval = 60000*.25)

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

            onset.append(time_subj_dict_all[task][subj][0] - shift_time_list[1]*60000)
            onset.extend(time_subj_dict_all[task][subj][:-1]) 
            offset.extend(time_subj_dict_all[task][subj])
            pred.extend(merged_pred_dict_all[task][subj])

            for df_ in task_to_storing_dict[task][subj]:
                subj_df = pd.concat([subj_df, df_])
            pred_df = pd.DataFrame(data = {'onset': onset, 'offset': offset, 'pred': pred})

            pred_df = merge_toy_pred(pred_df, subj_df)
            toy_pred_list[task][subj] = pred_df

    # if CHECKING:  
    #     for subj in subj_list:
    #         pred = []
    #         onset = []
    #         offset = []
    #         for task in tasks:
    #             onset.append(time_subj_dict_all[task][subj][0] - shift_time_list[1]*60000)
    #             onset.extend(time_subj_dict_all[task][subj][:-1]) 
    #             offset.extend(time_subj_dict_all[task][subj])
    #             pred.extend(merged_pred_dict_all[task][subj])
            
    #         df = pd.DataFrame({"onset":onset, "offset": offset, "pred": pred})
    #         df['pred'] = df['pred'].replace({4: "No_toys", 2: "F+", 0: "F", 3: "E", 1:"E+"})
            
    #         df = df.sort_values(by=['onset'])
    #         df.to_csv('./data/interim/prediction/20210727/no_ops_threshold_'+str(no_ops_theshold)+'/window_size_'+str(interval_length)+'/state_'+str(n_states)+'/'+str(subj)+'.csv', index = False)
    
    # # state_name_dict = {0: "No_toys", 3: "F+", 2: "F",  1: "E", 4:"E+"}

    for subj in subj_list:
        for task in tasks:
            if subj == 13 and task == 'NMS':
                print(time_subj_dict_all[task][subj])
                print(np.diff(time_subj_dict_all[task][subj]))
            df = pd.DataFrame()
            for df_ in task_to_storing_dict[task][subj]:
                df = pd.concat([df, df_])
            # df = task_to_storing_dict[task][subj].reset_index(drop = True)
            pred_state_list= merged_pred_dict_all[task][subj]
            state_name_list = [state_name_dict[s] for s in pred_state_list]
            time_list = time_subj_dict_all[task][subj]
            # if len(time_list) > 0 and len(pred_state_list) > 0:
                # print(len(time_list), len(pred_state_list) > 0)
            if CHECKING:
                fig_name = './figures/hmm/20210727/no_ops_theshold_'+str(no_ops_theshold)+'/window_size_1.5/'+str(n_states)+'_states/'+task+'/'+str(subj)+".png"
                draw_timeline_with_merged_states(subj, df, pred_state_list, time_list, state_name_dict, fig_name= fig_name, gap_size = shift_time_list[1], show=False)
                # plain_fig_name = './figures/hmm/20210721/feature_discretization/'+task+'/'+str(subj)+".png"
                # plain_fig_name = './figures/hmm/20210721/plain_timeline/'+task+'/'+str(subj)+".png"

                # draw_plain_timeline(subj, df, time_list,plain_fig_name)
                # save_png(fig, './figures/hmm/20210721/window_size_1.5/'+str(n_states)+'_states/'+task+'/'+str(subj)+".png", 1600, 800)




    with open('./data/interim/20210727_no_ops_theshold_'+str(no_ops_theshold)+'_'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'wb+') as f:
        pickle.dump(merged_pred_dict_all, f)

    with open('./data/interim/20210727_no_ops_theshold_'+str(no_ops_theshold)+'_'+str(n_states)+'_states_merged_prediction_prob_'+str(interval_length)+'_min.pickle', 'wb+') as f:
        pickle.dump(merged_proba_dict_all, f)
    
    with open('./data/interim/20210727_no_ops_theshold_'+str(no_ops_theshold)+'_'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'wb+') as f:
        pickle.dump(time_subj_dict_all, f)
    
    with open('./data/interim/20210727_no_ops_theshold_'+str(no_ops_theshold)+'_'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'wb+') as f:
        pickle.dump(toy_pred_list, f)

        
        
        
       