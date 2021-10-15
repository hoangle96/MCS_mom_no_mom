import numpy as np 
import pandas as pd 
import pickle
from variables import toys_dict, tasks, stationary_toys_list, mobile_toys_list, toy_colors_dict
import cv2
from pathlib import Path 
import glob 
import pandas as pd 
from visualization import draw_comparison, draw_timeline_with_prob_to_check, draw_toy_state, draw_toy_state_with_std
from pathlib import Path
from all_visualization_20210824 import rank_state

no_ops_time = 10
n_states = 5
feature_set = 'n_new_toy_ratio'
interval_length = 1.5
shift = .5
with open('./data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_all_prob_'+str(interval_length)+'_min.pickle', 'rb') as f:
    all_proba_dict = pickle.load(f)

with open('./data/interim/20210907'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold_'+str(n_states)+'_states_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
    pred_dict = pickle.load(f)
with open("./data/interim/20210907_"+str(no_ops_time)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
    time_arr_dict = pickle.load(f)

with open('./data/interim/20210824_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
    task_to_storing_dict = pickle.load(f)
from merge import merge_segment_with_state_calculation_all, merge_toy_pred
merged_pred_dict_all = {}
merged_proba_dict_all = {}
time_subj_dict_all = {}
all_prob_dict_all = {}
subj_list = list(task_to_storing_dict['MPS'].keys())
shift_time_list = np.arange(0,interval_length, shift)

for task in tasks:
    print(task)
    merged_df_dict = task_to_storing_dict[task]
    time_arr_shift_dict = time_arr_dict[task]
    pred_subj_dict = pred_dict[task]
    prob_subj_dict = all_proba_dict[task]

    merged_pred_dict_all_task_specific, merged_proba_dict_all_task_specific, time_subj_dict_all_task_specific, all_prob = merge_segment_with_state_calculation_all(subj_list, shift_time_list, merged_df_dict, time_arr_shift_dict, pred_subj_dict, prob_subj_dict, window_size = interval_length, n_states = n_states, shift_interval = 60000*shift)

    merged_pred_dict_all[task] = merged_pred_dict_all_task_specific
    merged_proba_dict_all[task] = merged_proba_dict_all_task_specific
    time_subj_dict_all[task] = time_subj_dict_all_task_specific
    all_prob_dict_all[task] = all_prob

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
# with open('./data/interim/20210816_30s_offset_new_merge_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    # pickle.dump(merged_pred_dict_all, f)
# with open('./data/interim/20210816_30s_offset_new_merge_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'wb+') as f:
    # pickle.dump(time_subj_dict_all, f)
model_file_name = "model_20210907_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
model_file_path = Path('./models/hmm/20210907/'+feature_set)/model_file_name
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)
state_name_dict = rank_state(model)

for subj in subj_list:
    for task in tasks:
        path = Path('./figures/hmm/20210907/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'_new_merge/'+str(n_states)+'_states/merged/'+task+'/')
        path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame()
        for df_ in task_to_storing_dict[task][subj]:
            df = pd.concat([df, df_])
        pred_state_list= merged_pred_dict_all[task][subj]
        state_name_list = [state_name_dict[s] for s in pred_state_list]
        time_list = time_subj_dict_all[task][subj]
        prob_list = all_prob_dict_all[task][subj]
        fig_name = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'_new_merge/'+str(n_states)+'_states/merged/'+task+'/'+str(subj)+".png"
        draw_timeline_with_prob_to_check(k = str(subj) + "window size: " + str(interval_length) + " no ops threshold "+ str(no_ops_time), \
                                        df = df, state_list = state_name_list, time_list = time_list,\
                                        state_name = state_name_dict, fig_name= fig_name, gap_size = shift, state_color_dict= state_color_dict,\
                                        prob_list = prob_list, shift = shift)

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

stationary_dict_for_std = {}
for state in state_name_dict.values():
    stationary_dict_for_std[state] = {}
    for toy in stationary_toys_list + ['no_toy']:
        stationary_dict_for_std[state][toy] = []

stationary_df = pd.DataFrame()
for task in ['MPS', "NMS"]:
    for subj in subj_list:
        if subj in toy_pred_list[task].keys():
            df_ = toy_pred_list[task][subj].copy()
            df_ = df_.explode('toys') 
            df_['toys'] = df_['toys'].replace({'no_ops':'no_toy'})

            df_['duration'] = df_['offset'] - df_['onset'] 
            df_['pred'] = df_['pred'].replace(state_name_dict)
            subj_stationary_dict = (df_.groupby(['pred', 'toys'])['duration'].sum()/df_.groupby(['pred'])['duration'].sum()).to_dict()
            for state in state_name_dict.values():
                for toy in stationary_toys_list+ ['no_toy']:
                    key = (state, toy)
                    if key in subj_stationary_dict.keys():
                        stationary_dict_for_std[state][toy].append(subj_stationary_dict[key])

        stationary_df = pd.concat([stationary_df,  toy_pred_list[task][subj]])
stationary_df = stationary_df.explode('toys') 
stationary_df['toys'] = stationary_df['toys'].replace({'no_ops':'no_toy'})

stationary_df['duration'] = stationary_df['offset'] - stationary_df['onset'] 
stationary_df['pred'] = stationary_df['pred'].replace(state_name_dict)
stationary_toy_to_pred_dict = (stationary_df.groupby(['pred', 'toys'])['duration'].sum()/stationary_df.groupby(['pred'])['duration'].sum()).to_dict()
stationary_toy_list = stationary_df['toys'].dropna().unique()

stationary_std = {}
for state in state_name_dict.values():
    stationary_std[state] = {}
    for toy in stationary_toys_list+ ['no_toy']:
        key = (state, toy)
        if key in stationary_toy_to_pred_dict.keys():
            stationary_std[state][toy] = np.abs(np.sum(np.array(stationary_dict_for_std[state][toy])-stationary_toy_to_pred_dict[key]))/len(stationary_dict_for_std[state][toy])

name = "Both conditions, fine motor toys"
# fig_path = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/toy_state_stationary.png'
# draw_toy_state(state_name_dict, stationary_toy_to_pred_dict, toy_colors_dict= toy_colors_dict, toy_list =  stationary_toy_list, name = name, fig_path =  fig_path,  indv = False)

fig_path = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/toy_state_stationary_2.png'
draw_toy_state_with_std(state_name_dict, stationary_toy_to_pred_dict, std_dict = stationary_std, toy_colors_dict= toy_colors_dict, toy_list =  stationary_toy_list, name = name, fig_path =  fig_path,  indv = False)

mobile_df = pd.DataFrame()
mobile_dict_for_std = {}
for state in state_name_dict.values():
    mobile_dict_for_std[state] = {}
    for toy in mobile_toys_list + ["no_toy"]:
        mobile_dict_for_std[state][toy] = []

for task in ['MPM', "NMM"]:
    for subj in subj_list:
        if subj in toy_pred_list[task].keys():
            df_ = toy_pred_list[task][subj].copy()
            df_ = df_.explode('toys') 
            df_['toys'] = df_['toys'].replace({'no_ops':'no_toy'})

            df_['duration'] = df_['offset'] - df_['onset'] 
            df_['pred'] = df_['pred'].replace(state_name_dict)
            subj_mobile_dict = (df_.groupby(['pred', 'toys'])['duration'].sum()/df_.groupby(['pred'])['duration'].sum()).to_dict()
            for state in state_name_dict.values():
                for toy in mobile_toys_list+ ['no_toy']:
                    key = (state, toy)
                    if key in subj_mobile_dict.keys():
                        mobile_dict_for_std[state][toy].append(subj_mobile_dict[key])

            mobile_df = pd.concat([mobile_df,  toy_pred_list[task][subj]])
mobile_df = mobile_df.explode('toys') 
mobile_df['toys'] = mobile_df['toys'].replace({'no_ops':'no_toy'})
mobile_df['duration'] = mobile_df['offset'] - mobile_df['onset'] 
mobile_df['pred'] = mobile_df['pred'].replace(state_name_dict)
mobile_toy_to_pred_dict = (mobile_df.groupby(['pred', 'toys'])['duration'].sum()/mobile_df.groupby(['pred'])['duration'].sum()).to_dict()
mobile_toy_list = mobile_df['toys'].dropna().unique()

mobile_std = {}
for state in state_name_dict.values():
    mobile_std[state] = {}
    for toy in mobile_toys_list+ ['no_toy']:
        key = (state, toy)
        if key in mobile_toy_to_pred_dict.keys():
            mobile_std[state][toy] = np.abs(np.sum(np.array(mobile_dict_for_std[state][toy])-mobile_toy_to_pred_dict[key]))/len(mobile_dict_for_std[state][toy])


name = "Both conditions, gross motor toys"
# fig_path = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/toy_state_mobile.png'
# draw_toy_state(state_name_dict, mobile_toy_to_pred_dict, toy_colors_dict= toy_colors_dict, toy_list = mobile_toy_list, name = name, fig_path =  fig_path,  indv = False)
fig_path = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/toy_state_mobile_3.png'
draw_toy_state_with_std(state_name_dict, mobile_toy_to_pred_dict, toy_colors_dict= toy_colors_dict, toy_list = mobile_toy_list, name = name, fig_path =  fig_path,  indv = False, std_dict = mobile_std)


fig_name_by_task = {
'MPS': "With caregivers, fine motor toys",
'MPM': "With caregivers, gross motor toys",
'NMS': "Without caregivers, fine motor toys",
'NMM': "Without caregivers, gross motor toys",
}

for task in tasks:
    if task == "NMM" or task == 'MPM':
        toys_list = mobile_toys_list 
    elif task == 'NMS' or task == 'MPS':
        toys_list = stationary_toys_list 

    task_dict_for_std = {}
    for state in state_name_dict.values():
        task_dict_for_std[state] = {}
        for toy in toys_list+ ['no_toy']:
            task_dict_for_std[state][toy] = []

    df = pd.DataFrame()
    for subj in subj_list:
        if subj in toy_pred_list[task].keys():
            df_ = toy_pred_list[task][subj].copy()
            df_ = df_.explode('toys') 
            df_['toys'] = df_['toys'].replace({'no_ops':'no_toy'})

            df_['duration'] = df_['offset'] - df_['onset'] 
            df_['pred'] = df_['pred'].replace(state_name_dict)
            subj_dict = (df_.groupby(['pred', 'toys'])['duration'].sum()/df_.groupby(['pred'])['duration'].sum()).to_dict()
            for state in state_name_dict.values():
                for toy in toys_list+ ['no_toy']:
                    key = (state, toy)
                    if key in subj_dict.keys():
                        task_dict_for_std[state][toy].append(subj_dict[key])

            df = pd.concat([df, toy_pred_list[task][subj]])
    df = df.explode('toys') 
    df['toys'] = df['toys'].replace({'no_ops':'no_toy'})

    df['duration'] = df['offset'] - df['onset'] 
    df['pred'] = df['pred'].replace(state_name_dict)
    toy_to_pred_dict = (df.groupby(['pred', 'toys'])['duration'].sum()/df.groupby(['pred'])['duration'].sum()).to_dict()
    toy_list = np.sort(df['toys'].dropna().unique())
    std_dict = {}
    for state in state_name_dict.values():
        std_dict[state] = {}
        for toy in toys_list+ ['no_toy']:
            key = (state, toy)
            if key in toy_to_pred_dict.keys():
                std_dict[state][toy] = np.abs(np.sum(np.array(task_dict_for_std[state][toy])-toy_to_pred_dict[key]))/len(task_dict_for_std[state][toy])


    name = fig_name_by_task[task]
    # fig_path = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/'+task+'.png'
    # draw_toy_state(state_name_dict, toy_to_pred_dict = toy_to_pred_dict, toy_list = toy_list, toy_colors_dict = toy_colors_dict, name = name, fig_path= fig_path, indv = True)
    fig_path = './figures/hmm/state_distribution_20210815_30s/'+feature_set+'/no_ops_threshold_'+str(no_ops_time)+'/window_size_'+str(interval_length)+'/'+str(n_states)+'_states'+'/'+task+'_3.png'
    draw_toy_state_with_std(state_name_dict, toy_to_pred_dict = toy_to_pred_dict, toy_list = toy_list, toy_colors_dict = toy_colors_dict, name = name, fig_path= fig_path, indv = True, std_dict=std_dict)

## merge with locomotion
