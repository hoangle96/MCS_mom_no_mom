import cv2
import numpy as np 
import pickle
from variables import tasks, toys_of_interest_dict, condition_name
from pathlib import Path 
import glob 
import pandas as pd 
from visualization import draw_comparison

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

if __name__ == "__main__":
    task = "MPM"
    no_ops_threshold = 7
    interval_length =2
    toy_to_check = toys_of_interest_dict[task]
    check = 'both'
    pos_example = True

    if check == 'merging':
        with open('./data/interim/20210729_merged_container_only_clean_data_for_feature_engineering.pickle', 'rb') as f:
            clean_data_dict = pickle.load(f)
    elif check == 'removing':
        with open('./data/interim/20210729_remove_no_ops_only_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
            clean_data_dict = pickle.load(f)
    elif check == 'both':
        with open('./data/interim/20210804_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
            clean_data_dict = pickle.load(f)
    
    if "M" in task:
        k = 'bucket' # or grocery_cart
        toy_to_check = {k: toy_to_check[k]}
    
    for container, contained in toy_to_check.items():
        if isinstance(contained, list) or isinstance(contained, set):
            toys_of_interest_list = contained + [container]
            contained_tuple = tuple([contained])
            container_tuple = tuple([container])
        else:
            toys_of_interest_list = [container, contained]
            contained_tuple = tuple(contained)
            container_tuple = tuple([container])
    
    toy_of_interest_set = set(toys_of_interest_list)


    with open('./data/interim/20210729_clean_data_for_feature_engineering.pickle', 'rb') as f:
        clean_data_dict_og = pickle.load(f)
    
    # with open("./data/interim/20210729_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_"+str(interval_length)+"_min_2.pickle", 'rb') as f:
    #     feature_dict = pickle.load(f)

    # with open("./data/interim/20210729_"+str(no_ops_threshold)+"_no_ops_threshold_feature_engineering_time_arr_"+str(interval_length)+"_min_2.pickle", 'rb') as f:
    #     time_arr_dict = pickle.load(f)

    # with open("./data/interim/20210729_"+str(no_ops_threshold)+"_no_ops_threshold_label_"+str(interval_length)+"_min_2.pickle", 'rb') as f:
    #     labels_dict = pickle.load(f)
    
    # with open('./data/interim/20210729_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering_2.pickle', 'rb') as f:
    #     clean_data_dict = pickle.load(f)

    # get the 
    key_list = list(clean_data_dict_og[task].keys())

    # idx = np.random.randint(low = 0, high = len(list(feature_dict[task].keys())))

    # get the right dataframe

    not_sat = True
    while not_sat:
        idx = np.random.randint(low = 0, high = len(key_list))
        subj = key_list[idx]
        df_list = clean_data_dict[task][subj]

        df = pd.DataFrame()
        for df_ in df_list:
            df = pd.concat([df, df_])
        
        if "index" in df.columns:
            df = df.drop(columns=['index'])


        df_toys = df['toy'].to_numpy()
        # if pos_example:
        row_condition = [(set(i).issubset(toy_of_interest_set)) and ((set(container_tuple).issubset(set(i)))) for i in df_toys]
        # else:
        s = pd.Series(row_condition, name='bools')
        merged = 1 if pos_example == True else 0
        df_to_select = df[s.values]
        df_to_select = df_to_select.loc[df_to_select.loc[:,"merged_toy"] == merged, :]
        # print(df_to_select[['onset', 'offset', 'toy']])
        m = df_to_select.index.to_series().diff().ne(1).cumsum()
        df_to_select['consec_group'] = m
        # print(df_to_select)
        og_df = pd.DataFrame()
        for df_ in clean_data_dict_og[task][subj]:
            og_df = pd.concat([og_df, df_])

        for i in m:
            # print(i)
            df_extract = df_to_select.loc[df_to_select.loc[:,'consec_group'] == i, :]
            changed_toys = df_extract['toy'].tolist()
            changed_onset_list = df_extract['onset'].tolist()
            changed_offset_list = df_extract['offset'].tolist()

            onset = df_extract.iloc[0]['onset'].item()
            onset_minutes, onset_seconds, onset_rest = convert_to_min(onset)
            onset_time_in_databrary = str(onset_minutes)+":"+str(onset_seconds)+"."+str(onset_rest)

            offset = df_extract.iloc[-1]['offset'].item()
            offset_minutes, offset_seconds, offset_rest = convert_to_min(offset)
            offset_time_in_databrary = str(offset_minutes)+":"+str(offset_seconds)+"."+str(offset_rest)

            original_toy_df = og_df.loc[(og_df.loc[:,'onset'] >= onset) & (og_df.loc[:,'offset'] <= offset), :]
            original_toy_df_toys = original_toy_df['toy'].tolist()
            original_onset_list = original_toy_df['onset'].tolist()
            original_offset_list = original_toy_df['offset'].tolist()

            if pos_example:
                print("Positive Example")
            else:
                print("Negative Example")
            print("Subject: ", subj, "\nCondition: ", task, "\nMerged toys: ", toy_of_interest_set, "\nOnset: ", onset_time_in_databrary, "\nOffset: ", offset_time_in_databrary)
            title = "Subject: " +str(subj) + ", condition: "+ condition_name[task] + ". Merged toys: " + ", ".join(toy_of_interest_set) + ". Original (top), merged (bottom)"
            draw_comparison(subj, df_list = [og_df, df], title = title, roi_onset= onset, roi_offset=offset)
            new_toy_labels = []
            og_toy_labels = []
            for time_pt in np.arange(onset, offset+.1, 1000/30):
                # for i in zip(range(len(changed_toys)), changed_onset_list, changed_offset_list):
                    # print(i)
                changed_toy_idxes = []
                # for idx_, on, off in zip(range(len(changed_toys)), changed_onset_list, changed_offset_list):
                #     if time_pt >= on and time_pt <= off:
                #         changed_toy_idxes.append(idx_)
                # if len(changed_toy_idxes) == 0:
                #     print(time_pt)
                #     print(changed_onset_list)
                #     print(changed_offset_list)

                #     print('here')

                if time_pt <= offset:
                    changed_toy_idx_list = [idx_ for idx_, on, off in zip(range(len(changed_toys)), changed_onset_list, changed_offset_list) if time_pt >= on and time_pt <= off]
                    if len(changed_toy_idx_list) > 0:
                        changed_toy_idx = min(changed_toy_idx_list)
                        new_toy_labels.append(changed_toys[changed_toy_idx])
                    else:
                        new_toy_labels.append("")

                    original_toy_idx = min([idx_ for idx_, on, off in zip(range(len(original_toy_df_toys)), original_onset_list, original_offset_list) if time_pt >= on and time_pt <= off ])
                    og_toy_labels.append(original_toy_df_toys[original_toy_idx])


            if offset - onset > 4*60000:
                offset -= (offset - onset)/2
                og_toy_labels = og_toy_labels[:len(og_toy_labels)//2]
                new_toy_labels = new_toy_labels[:len(new_toy_labels)//2]

            read_write_example(subj, onset, offset, og_toy_labels, new_toy_labels)