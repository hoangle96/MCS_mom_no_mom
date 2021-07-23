import pandas as pd
import numpy as np
import glob
from feature_engineering import shift_signal_toy_count
from visualization import save_png, draw_plain_timeline
from pathlib import Path
from variables import toys_dict, tasks, toys_list, toys_of_interest_dict
import pickle 
import itertools
from feature_engineering import shift_signal_toy_count

def remove_small_no_ops(df, small_no_ops_threshold):
    """
    1. remove no_ops <= small_no_ops_threshold
    2. merge rows with the same toys
    """
    df = df.reset_index(drop=True)
    small_no_ops_idx = df.loc[(df.loc[:, 'toy'] == "no_ops") & (df.loc[:, 'offset'] - df.loc[:, 'onset'] <= small_no_ops_threshold), :].index.values.tolist()
    df = df.drop(index = small_no_ops_idx)
    df = df.reset_index(drop = True)
    df['g'] = df['toy'].ne(df['toy'].shift()).cumsum()

    for group in df['g'].unique():
        if len(df.loc[(df['g'] == group)]) > 1:
            df_ = df.loc[(df['g'] == group), :]
            idx = df_.index.tolist()[0]
            last_idx = df_.index.tolist()[-1]
            df.iloc[idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
            df = df.drop(index=range(idx+1, last_idx+1))
            df = df.reset_index(drop=True)
    df =  df.drop(columns = 'g')
    return df

def get_consecutive_toys(df, toys_of_interest):
    df = df.reset_index(drop=True)

    for container, contained in toys_of_interest.items():
        proceed = False

        if isinstance(contained, list) or isinstance(contained, set):
            all_cols_condition = []
            all_cols_condition.append(df[container+"_ord"].notna().any())
            for t in contained:
                all_cols_condition.append(df[t+"_ord"].notna().any())
            proceed = np.any(all_cols_condition)
            toys_of_interest_list = []
            toys_of_interest_list.append(container)
            toys_of_interest_list.extend(contained)
            # toys_of_interest_list = np.array(toys_of_interest_list)

        else:
            proceed = df[container+"_ord"].notna().any() and df[contained+"_ord"].notna().any()
            toys_of_interest_list = [container, contained]

        if proceed:
            df = df.reset_index()
            # print(df)
            df['contain'] = 0
                
            # check to see if the row contains the toys we care about
            df_toy_np = df.toy.tolist()
            # condition: the toys in the row are a subset of proper subset of the toys of interest
            toy_of_interest_set = set(toys_of_interest_list)

            row_condition = [set(i).issubset(toy_of_interest_set) for i in df_toy_np]

            df.loc[row_condition, 'contain'] = 1

            # group the consecutive rows that have the toy we care together
            df['g'] = df['contain'].ne(df['contain'].shift()).cumsum()

            for group in df['g'].unique():
                if len(df.loc[(df['g'] == group)&(df['contain'] ==1), :]) > 1:
                    toy_playing_now = df.loc[(df['g'] == group)&(df['contain'] ==1), 'toy'].to_numpy().tolist()
                    toy_playing_now = list(set(itertools.chain.from_iterable(toy_playing_now)))

                    df_ = df.loc[(df['g'] == group)&(df['contain'] ==1), :]
                    # print(df_)
                    idx = df_.index.tolist()[0]
                    last_idx = df_.index.tolist()[-1]
                    df.iloc[idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
                    df.at[idx, 'toy'] = toy_playing_now
                    df = df.drop(index=range(idx+1, last_idx+1))
                    df = df.reset_index(drop=True)
                    # print(df.loc[(df['g'] == group), :])
                    # print(df)
            print("works")
    return df

def add_toy_to_single_col(row: pd.Series, toy_list: list) -> list:
    """
    Checking if the infant interacts with each toy by iterating through the columns and check if the column containing the toy name is NaN.
    Call this function by using df.apply(axis = 1)

    Parameters:
    -----------
        row: each row in the dataframe
        toy_list: the list of all toys available in the room (usually there are 13 toys)

    Returns:
    --------
        val: list of toys the infant plays with at that instance
    """
    val = []
    for toy in toy_list:
        col_name = 'all_toys.'+toy+"_ord"
        if pd.notna(row[col_name]):
            val.append(toy)

    if len(val) == 0:
        val = [["no_ops"]]
    return val

def insert_no_ops(df, onset_col, toy_list, task_onset, task_offset):
    # print(df.head())
    prev_offset = df[onset_col][0]
    new_col = []

    for col in df.columns:
        if 'all_toys' in col:
            col = col.split('.')[-1]
        new_col.append(col)
    df.columns = new_col

    for idx, row in enumerate(df.itertuples()):
        # print(row)
        if idx > 0:
            if row.onset != prev_offset:
                # insert stuff
                data = {'onset': prev_offset,
                        'offset': row.onset, 'toy': ["no_ops"]}
                for toy in toy_list:
                    data[toy+"_ord"] = np.nan
                line = pd.DataFrame(data=data)
                df = pd.concat([df.iloc[:idx], line, df.iloc[idx:]]).reset_index(drop=True)
            prev_offset = row.offset

    first_row = df.iloc[0, :]
    last_row = df.iloc[-1, :]
    start_time = first_row['onset']
    end_time = last_row['offset']
    
    # if exclude is not None:
        # task_onset 
    if start_time > task_onset:
        data = {'onset': task_onset,
                        'offset': start_time, 'toy': ["no_ops"]}
        for toy in toy_list:
            data[toy+"_ord"] = np.nan
        line = pd.DataFrame(data=data)
        df = pd.concat([line, df]).reset_index(drop=True)
    if end_time < task_offset:
        data = {'onset': end_time,
                        'offset': task_offset, 'toy': ["no_ops"]}
        for toy in toy_list:
            data[toy+"_ord"] = np.nan
        line = pd.DataFrame(data=data)
        df = pd.concat([df, line]).reset_index(drop=True)
    df = df.sort_values(by=['onset'])


    df = df.reset_index(drop=True)
    return df

# def compare_task_time_with_floortime(df, task_onset: int, task_offset: int, floor_time_onset: list, floor_time_offset: list):
#     exclude = []
#     new_exclusion = []
#     for idx, f_t_onset in enumerate(floor_time_onset):
        
#         if f_t_onset == task_onset and task_offset > floor_time_offset[idx]:
#             # each_task_df =  pd.concat([each_task_df, df.loc[(df.loc[:, 'all_toys.onset'] >= task_onset) & (df.loc[:,  'all_toys.onset'] <= floor_time_offset[idx]), :].reset_index()])
#             exclude.append(floor_time_offset[idx])
#             if 
#         elif f_t_onset > task_onset and task_offset == floor_time_offset[idx]:
#             # each_task_df =  pd.concat([each_task_df,  df.loc[(df.loc[:, 'all_toys.onset'] >= f_t_onset) & (df.loc[:,  'all_toys.onset'] <= task_offset), :].reset_index()])
#             exclude.append([f_t_onset, task_onset])

#     if len(exclude) == 0:
#         exclude = None

#     return exclude

if __name__ == '__main__':
    # pre-defined vars
    CHECKING = True # if True will export the CSV and draw out the interaction timeline

    mps_dict = {}
    mpm_dict = {}
    nms_dict = {}
    nmm_dict = {}

    task_to_storing_dict = {'MPS': mps_dict,
                        'MPM': mpm_dict, 'NMS': nms_dict, 'NMM': nmm_dict}
    
    toy_names = {str(v): k for k, v in toys_dict.items()}

    subj_dict = {}

    data_path = './data/raw/*.csv'
    floor_time = {}
    for file_name in glob.glob(data_path):
        print(file_name)
        subj = int(file_name.split('/')[-1].split('.')[0])
        # if subj == 3:

        print(subj)
        # if subj != 39 and subj != 4 or (subj != 43 and task != 'NMM'):
        # if subj == 3:
        #     print('here')
        df = pd.read_csv(file_name)
        # 20210721: floor_time indicates the parts of the sessions that were in included in Justine's analysis
        floor_time_onset = df['floortime.onset'].dropna().values
        floor_time_offset = df['floortime.offset'].dropna().values
        # print(floor_time_onset, floor_time_offset)

        subj_task_onset_offset = {}
        floor_time[subj] = {}

        for task in tasks:
            if subj == 26 and task == 'MPS':
                print('here')
            subj_dict[task] = {}
            floor_time[subj][task] = {}

            task_onset_offset = df.loc[df.loc[:, 'task.task'] == task, ['task.onset', 'task.offset']].to_numpy().flatten()
            task_onset, task_offset = task_onset_offset[0], task_onset_offset[1]
            
            subj_task_onset_offset[task] = [task_onset, task_offset]

            current_task_floor_time = []
            df_ = pd.DataFrame()
            df_list = []
            for idx, f_time_onset in enumerate(floor_time_onset):
                if f_time_onset >= task_onset and floor_time_offset[idx] <= task_offset:
                    current_task_floor_time.append([f_time_onset, floor_time_offset[idx]])
                    f_time_offset = floor_time_offset[idx]
                    each_task_df = df.loc[(df.loc[:,  'all_toys.onset'] >= f_time_onset) & (df.loc[:,  'all_toys.offset'] <= floor_time_offset[idx]), :].reset_index()
                    # exclude = compare_task_time_with_floortime(df.copy(), task_onset, task_offset, floor_time_onset, floor_time_offset)
                    # print(each_task_df.columns)
                    each_task_df = each_task_df.drop(columns='all_toys.ordinal')
                    cols = [col for col in each_task_df.columns if 'all_toys' in col]
                    each_task_df['toy'] = each_task_df.apply(add_toy_to_single_col, axis=1, args=(toys_list,))
                    each_task_df.columns = each_task_df.columns.str.replace(r'\s+', '')
                    each_task_df = insert_no_ops(each_task_df[cols+['toy']], 'all_toys.offset', toys_list, f_time_onset, f_time_offset)
                    # print(each_task_df.columns)
                    each_task_df = remove_small_no_ops(each_task_df, small_no_ops_threshold = 3000)
                    # print(df_.columns)
                    each_task_df = get_consecutive_toys(each_task_df, toys_of_interest_dict[task])
                    df_ = pd.concat([df_, each_task_df])
                    df_list.append(df_)

            # each_task_df = insert_no_ops(df_, 'offset', toys_list, task_onset, task_offset)
            df_ = df_.reset_index(drop = True)
            if len(df_) == 0:
                print('here')
            task_to_storing_dict[task][subj] = df_list
            floor_time[subj][task] = current_task_floor_time

            if CHECKING:
                # export the csv file
                df_check = pd.DataFrame()
                for df_ in df_list:
                    df_check = pd.concat([df_check, df_])
                df_check.to_csv('./data/interim/'+str(task)+'/'+str(subj)+'.csv', index = False)

                # plot out the timestep toy interaction to check
                plain_fig_name = './figures/hmm/20210721/plain_timeline/'+task+'/'+str(subj)+".png"
                draw_plain_timeline(subj, df_check, plain_fig_name)


    with open('./data/interim/20210721_clean_data_for_feature_engineering.pickle', 'wb+') as f:
        pickle.dump(task_to_storing_dict, f)
    with open('./data/interim/20210721_floor_time.pickle', 'wb+') as f:
        pickle.dump(floor_time, f)