import pandas as pd
import numpy as np
import glob
from feature_engineering import shift_signal_toy_count
from merge import merge_segment_with_state_calculation_all
from visualization import save_png, draw_plain_timeline, draw_comparison_new_merge, draw_plain_timeline_with_feature_discretization_compare
from pathlib import Path
from variables import tasks, condition_name
import pickle 
import itertools
from feature_engineering import shift_signal_toy_count
import cv2
import time
from math import ceil

def read_write_example(subj, onset, offset, og_labels, changed_labels):
    fps = 30
    first_frame = (onset/1000)*30
    last_frame = (offset/1000)*30
    n_frame = int(last_frame-first_frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    vid_file_path = './vid/*.mp4'

    for file_name in glob.glob(vid_file_path):
        subj_num = int(file_name.split("/")[-1].split("_")[1])
        # print(subj_num)
        if subj_num == subj:
            # f = # put here the frame from which you want to start
            cap = cv2.VideoCapture(file_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))
            ret, frame = cap.read()
            i = 0
            all_frames = []
            while ret and i <= n_frame:
                all_frames.append(frame)
                ret, frame = cap.read()
                i += 1
            
            cap.release()
            
            height, width, _ = all_frames[0].shape
            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter('./examples/outpy.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (width,height))
            
            for idx, frame in enumerate(all_frames):
                cv2.putText(frame, 
                "Orginal annotation: "+ str(og_labels[idx]), 
                (width//8, height-80), 
                font, 2, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4)

                cv2.putText(frame, 
                "New annotation: "+ str(changed_labels[idx]), 
                (width//8, height-20), 
                font, 2, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)

                out.write(frame)
            
            out.release()
            cv2.destroyAllWindows()


def convert_to_min(milli):
    minutes=int((milli/(1000*60))%60)
    seconds=int((milli/1000)%60)
    rest = milli - minutes*1000*60 - seconds*1000
    return minutes, seconds, int(rest)


if __name__ == '__main__':
    task = "MPM"
    interval_length = 2
    with open('./data/interim/20210805_floor_time.pickle', 'rb') as f:
        floor_time = pickle.load(f)
    
    with open('./data/interim/20210729_clean_data_for_feature_engineering.pickle', 'rb') as f:
        clean_data_dict_og = pickle.load(f)

    data = {}
    feature_all_dict = {}
    time_list_all_dict = {}
    y_labels = ["Original"]
    for no_ops_threshold in [5, 7, 10]:
        with open('./data/interim/20210805_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
            clean_data_dict = pickle.load(f)
            data[no_ops_threshold] = clean_data_dict    

        with open("./data/interim/20210805_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min.pickle", 'rb') as f:
            feature_dict = pickle.load(f)
            feature_all_dict[no_ops_threshold] = feature_dict

        with open("./data/interim/20210805_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min.pickle", 'rb') as f:
            time_arr_dict = pickle.load(f)
            time_list_all_dict[no_ops_threshold] = time_arr_dict
        
        y_labels.append(str(no_ops_threshold) + "s")

    key_list = list(clean_data_dict_og["MPS"].keys())
    not_sat = True
    # print(time.clock(), ceil(time.clock()%1*10))
    np.random.seed(ceil(time.clock()%1*10))
    while not_sat:
        idx = np.random.randint(low = 0, high = len(key_list))
        # idx = 13
        # subj = key_list[idx]
        subj = 13
        all_df_list = []

        og_df = pd.DataFrame()
        for df_ in clean_data_dict_og[task][subj]:
            og_df = pd.concat([og_df, df_])
        
        all_df_list.append(og_df)
        for no_ops_threshold in [5, 7, 10]: 
            df_list = data[no_ops_threshold][task][subj]

            df = pd.DataFrame()
            for df_ in df_list:
                df = pd.concat([df, df_])
            
            if "index" in df.columns:
                df = df.drop(columns=['index'])
            # df = df.loc[df.loc[:,"merge"] == 1, :]
            # print(df)
            all_df_list.append(df)
        print("Subject: ", subj, "\nCondition: ", condition_name[task])
        title = "Subject: " +str(subj) + ", condition: "+ condition_name[task] + ". From top down: Original, 5, 7, 10s threshold"
        draw_comparison_new_merge(str(subj) + "_"+task, df_list = all_df_list, y_labels = y_labels, title = title)

        # time_big_list = [time_list_all_dict[i][task][subj][0] for i in [5, 7, 10]] 
        # features_big_list = [feature_all_dict[i][task][subj][0] for i in [5, 7, 10]] 
        # gap_size = interval_length
        # fav_toy_big_list = [None, None, None]
        # fig_name = './examples/'+str(subj)+'_features.png'
        # draw_plain_timeline_with_feature_discretization_compare(str(subj) + "_"+task+"_features", all_df_list[1:], time_big_list, features_big_list, gap_size, fav_toy_big_list, fig_name)
        

    

        # subj_list = list(clean_data_dict['MPS'].keys())
        # for subj in subj_list:
        #     df_check = pd.DataFrame()
        #     for task in tasks:

        #         df_list = clean_data_dict[task][subj]
        #         for df_ in df_list:
        #             df_check = pd.concat([df_check, df_])
        #         plain_fig_name = './figures/hmm/20210729/plain_timeline_3/no_ops_threshold_'+str(no_ops_threshold)+'/'+task+'/'+str(subj)+".png"

        #         draw_plain_timeline(str(subj) + ', condition: ' + condition_name[task] +" threshold: " + str(no_ops_threshold), pd.concat(df_list), plain_fig_name)
        #     df_check = df_check[['onset', 'offset', 'toy']]
        #     df_check.to_csv('./data/to_check/check_reading_data_20210805/no_ops_threshold_'+str(no_ops_threshold)+'/'+str(subj)+'.csv', index = False)

    