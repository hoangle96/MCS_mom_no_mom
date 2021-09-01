import numpy as np, pickle
from variables import toys_dict, tasks, toys_list
import pandas as pd
# from visualization import draw_timeline_with_feature, save_png, draw_timeline_with_states

def merge_movement(pred_df, motion_df):
    time_list = []
    steps_list = []

    for row in pred_df.itertuples(index=False):
        # print(row)
        left_pt, right_pt = row.onset, row.offset
        overlap_df = motion_df.loc[((motion_df.loc[:,'babymovementOnset']<= right_pt) & (motion_df.loc[:,'babymovementOnset']>=left_pt)) 
                        | ((motion_df.loc[:,'babymovementOffset']<= right_pt) & (motion_df.loc[:,'babymovementOffset']>=left_pt)) 
                        | ((motion_df.loc[:,'babymovementOffset'] >= right_pt) & (motion_df.loc[:,'babymovementOnset'] <= left_pt)) ,:]
        total_time, total_steps = 0, 0

        if len(overlap_df) != 0:
            for overlap_row in overlap_df.itertuples(index=False):
                if overlap_row.babymovementOffset >= right_pt and overlap_row.babymovementOnset <= left_pt:
                    total_time +=  right_pt - left_pt
                    total_steps +=  (overlap_row.babymovementSteps/(overlap_row.babymovementOffset - overlap_row.babymovementOnset))*total_time
                else:
                    if overlap_row.babymovementOffset - overlap_row.babymovementOnset > 0:
                        avg_step = (int(overlap_row.babymovementSteps)/(overlap_row.babymovementOffset - overlap_row.babymovementOnset))
                        if overlap_row.babymovementOnset >= left_pt and overlap_row.babymovementOffset  <= right_pt:
                            total_time +=  overlap_row.babymovementOffset - overlap_row.babymovementOnset
                            total_steps += int(overlap_row.babymovementSteps)

                        elif overlap_row.babymovementOffset >= left_pt and overlap_row.babymovementOffset <= right_pt and overlap_row.babymovementOnset < left_pt:

                            total_time += (overlap_row.babymovementOffset - left_pt)
                            total_steps += avg_step*total_time

                        elif overlap_row.babymovementOnset  >= left_pt and overlap_row.babymovementOnset  <= right_pt and overlap_row.babymovementOffset  > right_pt:
                            time = right_pt - overlap_row.babymovementOnset
                            total_time += time
                            total_steps += avg_step*time
                    else:
                        total_steps += int(overlap_row.babymovementSteps)
        time_list.append(total_time) 
        steps_list.append(round(total_steps)) 

    pred_df['movement_time'] = time_list
    pred_df['steps'] = steps_list

    return pred_df

if __name__ == '__main__':
    with open('./data/interim/20210718_babymovement.pickle', 'rb') as f:
        babymovement_dict = pickle.load(f)

    for feature_set in ['n_new_toy_ratio', 'fav_toy_till_now', 'n_new_toy_ratio_and_fav_toy_till_now', 'new_toy_play_time_ratio']:
        for no_ops_time in [10, 5, 7]:
            print('no_ops_time', no_ops_time)
            for interval_length in [1.5, 1, 2]:
                for n_states in range(5, 7):
                    # with open('./data/interim/20210727_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    #     merged_pred_dict_all = pickle.load(f)

                    # with open('./data/interim/20210727_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
                    #     time_subj_dict_all = pickle.load(f)

                    # with open('./data/interim/20210805_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        # merged_pred_dict_all = pickle.load(f)

                    with open('./data/interim/20210816_30s_offset_new_merge_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        merged_pred_dict_all = pickle.load(f)

                    with open('./data/interim/20210816_30s_offset_new_merge_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_time_arr_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
                        time_subj_dict_all = pickle.load(f)

                
                    shift_time_list = np.arange(0, interval_length, .25)
                    merged_pred_w_locomotion = {}
                    for task in tasks:
                        merged_pred_w_locomotion[task] = {}
                        pred_by_task_dict = merged_pred_dict_all[task]
                        time_by_task_dict = time_subj_dict_all[task]
                        babymovement_by_task_dict = babymovement_dict[task]
                        for subj, pred in pred_by_task_dict.items():
                            # print(subj, task)
                            
                            onset = []
                            onset.append(time_subj_dict_all[task][subj][0] - shift_time_list[1]*60000)
                            onset.extend(time_subj_dict_all[task][subj][:-1]) 
                            # if task == "MPS" and 4 in pred_by_task_dict[subj]:
                                # print(subj)

                            df = pd.DataFrame(data = {'onset': onset, 'offset':time_by_task_dict[subj], 'pred': pred_by_task_dict[subj]})
                            # if subj == 39 and task == 'NMM':
                            #     print(df)
                            merged_pred_w_locomotion[task][subj] = merge_movement(df, babymovement_by_task_dict[subj])
                    # print(merged_pred_w_locomotion)
                    with open('./data/interim/20210816_30s_offset_new_merge_'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_locomotion_'+str(interval_length)+'_min.pickle', 'wb+') as f:
                        pickle.dump(merged_pred_w_locomotion, f)