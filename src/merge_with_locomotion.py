import numpy as np, pickle
from variables import toys_dict, tasks, toys_list
import pandas as pd
# from visualization import draw_timeline_with_feature, save_png, draw_timeline_with_states

def merge_movement(pred_df, motion_df):
    # key_tuple = tuple(prediction_dict.keys())
    # for k in key_tuple:
    # pred_df['left_bound'] = pred_df['timestep'].shift()
    # pred_df['left_bound'].fillna(pred_df['timestep'] - 120000, inplace = True) #switch to 120000 for unshifted 
    # pred_df.loc[:,'prob'] = pred_df.loc[:,'prob'].round(2)
    time_list = []
    steps_list = []

    for row in pred_df.itertuples(index=False):
        # print(row)
        left_pt, right_pt = row.onset, row.offset
        overlap_df = motion_df.loc[((motion_df.loc[:,'babymovementOnset']<= right_pt) & (motion_df.loc[:,'babymovementOnset']>=left_pt)) 
                        | ((motion_df.loc[:,'babymovementOffset']<= right_pt) & (motion_df.loc[:,'babymovementOffset']>=left_pt)) 
                        | ((motion_df.loc[:,'babymovementOffset'] >= right_pt) & (motion_df.loc[:,'babymovementOnset'] <= left_pt)) ,:]
        # print(overlap_df)
        # print(row)
        total_time, total_steps = 0, 0

        if len(overlap_df) != 0:
            for overlap_row in overlap_df.itertuples(index=False):
                # print(row)
                # print(overlap_row)
                if overlap_row.babymovementOffset >= right_pt and overlap_row.babymovementOnset <= left_pt:
                    total_time +=  right_pt - left_pt
                    total_steps +=  (overlap_row.babymovementSteps/(overlap_row.babymovementOffset - overlap_row.babymovementOnset))*total_time
                else:
                    if overlap_row.babymovementOffset - overlap_row.babymovementOnset > 0:
                        avg_step = (int(overlap_row.babymovementSteps)/(overlap_row.babymovementOffset - overlap_row.babymovementOnset))
                        if overlap_row.babymovementOnset >= left_pt and overlap_row.babymovementOffset  <= right_pt:
                            total_time +=  overlap_row.babymovementOffset - overlap_row.babymovementOnset
                            total_steps += int(overlap_row.babymovementSteps)

                            # total_distance += overlap_row.distance
                            # total_displace += overlap_row.displace
                            # total_steps += overlap_row.steps
                        elif overlap_row.babymovementOffset >= left_pt and overlap_row.babymovementOffset <= right_pt and overlap_row.babymovementOnset < left_pt:

                            total_time += (overlap_row.babymovementOffset - left_pt)
                            total_steps += avg_step*total_time

                            # total_area += overlap_row.area_per_sec*time
                            # total_distance += overlap_row.distance_per_sec*time
                            # total_displace += overlap_row.displace_per_sec*time
                            # total_steps += overlap_row.steps_per_sec*time
                        elif overlap_row.babymovementOnset  >= left_pt and overlap_row.babymovementOnset  <= right_pt and overlap_row.babymovementOffset  > right_pt:
                            time = right_pt - overlap_row.babymovementOnset
                            # total_area += overlap_row.area_per_sec*time
                            # total_distance += overlap_row.distance_per_sec*time
                            # total_displace += overlap_row.displace_per_sec*time
                            total_time += time
                            total_steps += avg_step*time
                    else:
                        total_steps += int(overlap_row.babymovementSteps)

            
        time_list.append(total_time) 
        steps_list.append(round(total_steps)) 


        # print()
    pred_df['movement_time'] = time_list
    pred_df['steps'] = steps_list

    return pred_df

if __name__ == '__main__':
    with open('./data/interim/20210721_5_states_merged_prediction_1.5_min.pickle', 'rb') as f:
        merged_pred_dict_all = pickle.load(f)
    
    with open('./data/interim/20210721_5_states_time_arr_dict_1.5_min.pickle', 'rb') as f:
        time_subj_dict_all = pickle.load(f)

    with open('./data/interim/20210718_babymovement.pickle', 'rb') as f:
        babymovement_dict = pickle.load(f)

    shift_time_list = [0,0.25,0.5,0.75,1]
    merged_pred_w_locomotion = {}
    # print(merged_pred_dict_all)
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
            if task == "MPS" and 4 in pred_by_task_dict[subj]:
                print(subj)

            df = pd.DataFrame(data = {'onset': onset, 'offset':time_by_task_dict[subj], 'pred': pred_by_task_dict[subj]})
            # if subj == 39 and task == 'NMM':
            #     print(df)
            merged_pred_w_locomotion[task][subj] = merge_movement(df, babymovement_by_task_dict[subj])
    print(merged_pred_w_locomotion)
    with open('./data/interim/20210721_merged_pred_w_locomotion_window_1.5_5_states.pickle', 'wb+') as f:
        pickle.dump(merged_pred_w_locomotion, f)