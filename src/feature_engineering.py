import collections
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.core.defchararray import index
import pandas as pd
from itertools import chain, groupby
import pickle
from variables import toys_dict, tasks, toys_list
from visualization import draw_timeline_with_feature, save_png, draw_plain_timeline_with_feature_discretization
from typing import List 
import itertools

def get_first_last_time(df):
    # Get the start and end time of the session
    first_row = df.iloc[0, :]
    last_row = df.iloc[-1, :]
    start_time = first_row['onset']
    end_time = last_row['offset']
    return start_time, end_time


def get_unique_toy(df):
    # get all unique toys
    toy_list = df.loc[:, 'toy'].tolist()
    toy_list_flatten = []
    for t in toy_list:
        if isinstance(t, list) or isinstance(t, set):
            if 'no_ops' not in t:
                toy_list_flatten.extend(t)
        elif t != 'no_ops':
            toy_list_flatten.append(t)

    toy_list_flatten_final = [t_ for t_ in toy_list_flatten if t_ != 'no_ops']
    unique_toys = np.unique(toy_list_flatten_final).tolist()
    return unique_toys


def rank_toy_local(df, unique_toys, left_pt, right_pt):
    """
    Find duration of play for each toy in the df
    unique toys: all the toys in the df
    left_pt: left bound of interval that needs calculation
    right_pt: right bound of interval that needs calculation
    Return
        dictionary of toys, ranked by playing time in this interval ascendingly 
    """

    toy_dict = {}
    for t in unique_toys:
        toy_dict[t] = 0

    # iterate through each row of the dataframe to add to the duration of the play
    for row in df.itertuples():
        # if the onset happens after the left_pt, the left_bound = onset, else left_pt itself
        # if the offset happens before the right_pt, the right_bound = offset, else right_pt itself

        left_bd = row.onset if row.onset >= left_pt else left_pt
        right_bd = row.offset if row.offset <= right_pt else right_pt
        duration = right_bd - left_bd
        if 'no_ops' not in row.toy:
            if isinstance(row.toy, list) or isinstance(row.toy, set):
                for t in row.toy:
                    toy_dict[t] += duration
            else:
                toy_dict[row.toy] += duration

    # ranked the toy by playing time, from least to most 
    ranked_toy_dict = {k: v for k, v in sorted(
        toy_dict.items(), key=lambda item: item[1])}
    return ranked_toy_dict


def get_new_toys(left_pt, right_pt, df, threshold):
    # going through all the toy and check for the last occurence. 
    # If the time difference between the last occurence and now is >= threshold then this toy is a "new" toy
    toy_cnt_df = get_current_segment(left_pt, right_pt, df)
    unique_toys = get_unique_toy(toy_cnt_df)

    new_toy_list = []
    for toy in unique_toys:
        this_toy = toy_cnt_df[[toy in x for x in toy_cnt_df.loc[:, 'toy']]]
        first_occurrence = this_toy.iloc[0]
        # print(first_occurrence)
        if first_occurrence['onset'] >= left_pt:
            # break
            all_toy = df[[toy in x for x in df.loc[:, 'toy']]]
            all_before = all_toy.loc[all_toy.loc[:, 'offset'] <= first_occurrence['onset'], 'offset']
            # print(all_before)
            if len(all_before) == 0:
                new_toy_list.append(toy)
            elif first_occurrence['onset'] - all_before.iloc[-1] >= threshold:
                new_toy_list.append(toy)
    return new_toy_list


def toy_encode(toy_list, big_toy_list):
    # return one-hot encoded vector of toylist
    return [1 if t in toy_list else 0 for t in big_toy_list]


def get_current_segment(left_pt, right_pt, df):
    return df.loc[((df.loc[:, 'onset'] <= right_pt) & (df.loc[:, 'onset'] >= left_pt))
                  | ((df.loc[:, 'offset'] <= right_pt) & (df.loc[:, 'offset'] >= left_pt))
                  | ((df.loc[:, 'offset'] >= right_pt) & (df.loc[:, 'onset'] <= left_pt)), :]


def get_feature_vector_toys_count(left_pt, right_pt, df, no_ops_threshold, new_toy_threshold, toys_list):

    # get all the interaction within the boundary
    curr = get_current_segment(left_pt, right_pt, df)

    # get all the no_ops < threshold
    small_no_ops = df.loc[(df.loc[:, 'offset'] <= right_pt) & (df.loc[:, 'offset'] >= left_pt) & (
        (df.loc[:, 'offset'] - df.loc[:, 'onset']) < no_ops_threshold) & (df.loc[:, 'toy'] == 'no_ops'), :]


    # segment length, usually == interval length but do the calculation for the tail end
    segment_len = (right_pt - left_pt)/60000
    n_episode = len(curr) - len(small_no_ops)

    n_switches = len(df.loc[(df.loc[:,'offset']<= right_pt) & (df.loc[:,'offset']>=left_pt),:]) - len(small_no_ops)

    # get all of the unique toys during the segment
    unique_toys = get_unique_toy(curr)

    # get all the toys ranked to find the favorite toy
    ranked_toy_dict = rank_toy_local(curr, unique_toys, left_pt, right_pt)
    if len(tuple(ranked_toy_dict.values())) == 0:
        fav_toy_duration_ratio = 0
    else:
        fav_toy_duration_ratio = tuple(
            ranked_toy_dict.values())[-1]/(right_pt-left_pt)

    begin_time, _ = get_first_last_time(df)
    toy_df_from_beginning = get_current_segment(begin_time, right_pt, df)

    ranked_toy_from_begin = rank_toy_local(
        toy_df_from_beginning, toys_list, begin_time, right_pt)

    dom_toy_till_current = tuple(ranked_toy_from_begin.keys())[-1]
    if dom_toy_till_current in ranked_toy_dict.keys():
        dom_toy_till_curr_duration_ratio = ranked_toy_dict[dom_toy_till_current]/(
            right_pt-left_pt)
    else:
        dom_toy_till_curr_duration_ratio = 0

    new_toys = get_new_toys(left_pt, right_pt, df, new_toy_threshold)
    n_new_toy = len(new_toys)

    return n_switches, unique_toys, n_new_toy, fav_toy_duration_ratio, dom_toy_till_curr_duration_ratio, dom_toy_till_current


def shift_signal_toy_count(shift_time, df, floor_time_begin, floor_time_end, interval_length, no_ops_threshold, new_toy_threshold, toys_list, CHECK = False):
    window_time = interval_length*60000
    no_ops_threshold = no_ops_threshold * 60000
    first_row = df.iloc[0, :]
    last_row = df.iloc[-1, :]
    # time_0 = first_row['onset']
    start_time = first_row['onset'] + shift_time*60000
    start_time = start_time if start_time >= floor_time_begin else floor_time_begin
    end_time = last_row['offset']
    end_time = end_time if end_time <= floor_time_end else floor_time_end

    ptr = start_time + window_time
    left_bound = start_time

    # all features to return

    time_arr = []

    toy_change_rate = []
    toy_per_min_list = []
    n_new_toy_list = []
    fav_toy_ratio_list = []
    dom_toy_till_curr_duration_ratio_list = []
    toy_iou = []

    # return these features for qc
    previous_toy_list = []
    toy_delta_list = []
    toy_present = []
    new_toy_list = []
    curr_dom_toy_list = []

    previous_toy_list = []
    i = 0
    # if shift_time != 0:
    #     ep_rate, toy_rate, toy_per_sc, unique_toys, toy_duration_ratio, n_new_toy_rate, dom_toy_till_curr_duration_ratio, curr_dom_toy = get_feature_vector_toys_count(time_0, start_time, df, no_ops_threshold, new_toy_threshold, toys_list)
    #     previous_toy_list = unique_toys

    while ptr < end_time or end_time - left_bound > window_time*1/2:
        if ptr > end_time:
            ptr = end_time

        
        n_toy_switches, unique_toys, n_new_toy, fav_toy_duration_ratio, fav_toy_till_curr_duration_ratio, fav_toy_till_current = get_feature_vector_toys_count(
            left_bound, ptr, df, no_ops_threshold, new_toy_threshold, toys_list)

        fav_toy_ratio_list.append(fav_toy_duration_ratio)
        n_new_toy_list.append(n_new_toy)

        toy_change_rate.append(n_toy_switches)
        toy_per_min_list.append(len(unique_toys))
        time_arr.append(ptr)
        curr_dom_toy_list.append(fav_toy_till_current)
        dom_toy_till_curr_duration_ratio_list.append(
            fav_toy_till_curr_duration_ratio)
        

        toy_present.append(toy_encode(unique_toys, toys_list))

        if i == 0 and shift_time == 0:
            toy_iou.append(0)
            v = toy_encode(unique_toys, toys_list)
        else:
            toy_insect = np.intersect1d(unique_toys, previous_toy_list)
            toy_union = np.union1d(unique_toys, previous_toy_list)
            if len(toy_union) == 0:
                toy_iou.append(0)
            else:
                toy_iou.append(len(toy_insect)/len(toy_union))

            toy_delta = np.setdiff1d(toy_union, toy_insect)
            v = toy_encode(toy_delta, toys_list)
        toy_delta_list.append(v)

        left_bound = ptr
        ptr += window_time
        previous_toy_list = unique_toys
        i += 1

    if CHECK:
        return time_arr, toy_change_rate, toy_per_min_list, n_new_toy_list, fav_toy_ratio_list, toy_iou, dom_toy_till_curr_duration_ratio_list, curr_dom_toy_list, toy_delta_list, toy_present
    else:
        return time_arr, toy_change_rate, toy_per_min_list, n_new_toy_list, fav_toy_ratio_list, toy_iou


if __name__ == '__main__':
    CHECK = False 
    # load data
    with open('./data/interim/20210721_clean_data_for_feature_engineering.pickle', 'rb') as f:
        task_to_storing_dict = pickle.load(f)

    with open('./data/interim/20210721_floor_time.pickle', 'rb') as f:
        floor_time = pickle.load(f)


    interval_length = 1
    no_ops_threshold = 5/60
    new_toy_threshold = 2
    # CHECKING =
    shift_time_list = np.arange(0, interval_length, .25)

    feature_dict = {}
    time_arr_dict = {}
    labels_dict = {}

    for task in tasks:
        feature_dict[task] = {}
        time_arr_dict[task] = {}
        labels_dict[task] = {}

        task_specfic_dict = task_to_storing_dict[task]
        # print(task_specfic_dict)
        for subj, df in task_specfic_dict.items():
            print(subj, task)
            if subj == 42 and task == 'NMS':
                print('here')
            feature_dict[task][subj] = {}
            time_arr_dict[task][subj] = {}
            labels_dict[task][subj] = {}
            
            for shift_time in shift_time_list:
                n_toy_switch = []
                n_toy_big_list =  []
                n_new_toy_big_list =  []
                fav_toy_ratio_big_list =  []
                time_arr_list = []

                for i in range(len(floor_time[subj][task])):
                    floor_time_begin, floor_time_end = floor_time[subj][task][i][0], floor_time[subj][task][i][1] 
                    time_arr, toy_change_rate, n_toy_list, n_new_toy_list, fav_toy_ratio_list, _, _, _, _, _ = shift_signal_toy_count(shift_time, df[i], floor_time_begin, floor_time_end, interval_length, no_ops_threshold, new_toy_threshold, toys_list, CHECK = True)
                    # print(time_arr)
                    n_toy_switch.extend(toy_change_rate)
                    n_toy_big_list.extend(n_toy_list)
                    n_new_toy_big_list.extend(n_new_toy_list)
                    fav_toy_ratio_big_list.extend(fav_toy_ratio_list)
                    time_arr_list.extend(time_arr)

                if len(time_arr_list) > 0:
                    feature = np.hstack((np.array(n_toy_switch).reshape((-1,1)),\
                                        np.array(n_toy_big_list).reshape((-1,1)),\
                                        np.array(n_new_toy_big_list).reshape((-1,1)),\
                                        np.array(fav_toy_ratio_big_list).reshape((-1,1))))
                                #   np.array(toy_iou_list).reshape((-1,1))))
                # print(feature)
                feature_dict[task][subj][shift_time] = feature
                time_arr_dict[task][subj][shift_time] = time_arr_list
                label =  np.where(feature[:,1] == 0, 0, None)
                labels_dict[task][subj][shift_time] = label
                if CHECK:
                    fig_name = './figures/hmm/20210721/feature_discretization/'+task+'/'+str(subj)+".png"

                    draw_plain_timeline_with_feature_discretization(subj, df, time_arr_list, fig_name)

    with open("./data/interim/20210721_feature_engineering_"+str(interval_length)+"_min.pickle", 'wb+') as f:
        pickle.dump(feature_dict, f)
    
    with open("./data/interim/20210721_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'wb+') as f:
        pickle.dump(time_arr_dict, f)
    
    with open("./data/interim/20210721_label_"+str(interval_length)+"_min.pickle", 'wb+') as f:
        pickle.dump(labels_dict, f)