import pandas as pd
import numpy as np
import glob

from pandas.core.reshape.merge import merge
from feature_engineering import shift_signal_toy_count
from merge import merge_segment_with_state_calculation_all
from visualization import save_png, draw_plain_timeline
from pathlib import Path
from variables import toys_dict, tasks, toys_list, toys_of_interest_dict, non_compatible_toys_dict, toy_to_task_dict, condition_name
import pickle 
import itertools
from feature_engineering import shift_signal_toy_count

def remove_small_no_ops(df, small_no_ops_threshold):
    """
    1. remove no_ops <= small_no_ops_threshold
    2. merge rows with the same toys
    """
    # print(df)
    no_ops_merge_cnt, no_ops_merge_time = 0 , 0
    df = df.reset_index(drop=True)
    all_toys = df['toy'].to_numpy()
    no_ops_row_condition = ['no_ops' in i for i in all_toys]
    small_no_ops_idx = df.loc[(no_ops_row_condition) & (df.loc[:, 'offset'] - df.loc[:, 'onset'] <= small_no_ops_threshold), :].index.values.tolist()
    df = df.drop(index = small_no_ops_idx)
    df = df.reset_index(drop = True)
    df['g'] = df['toy'].ne(df['toy'].shift()).cumsum()

    for group in df['g'].unique():
        if len(df.loc[(df['g'] == group)]) > 1:
            df_ = df.loc[(df['g'] == group), :]
            idx = df_.index.tolist()[0]
            last_idx = df_.index.tolist()[-1]
            df.iloc[idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
            df.iloc[idx, df.columns.get_loc('merge')] = 1
            no_ops_merge_cnt += 1
            no_ops_merge_time +=  df.iloc[last_idx, df.columns.get_loc('offset')] - df.iloc[idx, df.columns.get_loc('onset')] 
            df = df.drop(index=range(idx+1, last_idx+1))
            df = df.reset_index(drop=True)
    df = df.drop(columns = 'g')
    # print(df)

    return df, no_ops_merge_cnt, no_ops_merge_time

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
        col_name = 'all_toys_'+toy+"_ord"
        if pd.notna(row[col_name]):
            val.append(toy)

    if len(val) == 0:
        val = ["no_ops"]
    return val

def insert_no_ops(df, offset_col, toy_list, task_onset, task_offset, merging = False):
    # print(df.columns)
    # print(len(df.columns))

    prev_offset = df[offset_col][0]
    new_col = []
    # all_cols = df.columns

    # for t in toy_list:
    #     if t in all_cols:

    for col in df.columns:
        if 'all_toys' in col:
            col = "_".join(col.split('_')[2:])
        new_col.append(col)

    # new_col.append('toy')
    df.columns = new_col

    for idx, row in enumerate(df.itertuples()):
        # print(row)
        if idx > 0:
            if row.onset != prev_offset:
                # insert stuff
                data = {'onset': prev_offset, 'offset': row.onset, 'toy': "no_ops"}
                for toy in toy_list:
                    data[toy+"_ord"] = np.nan
                if merging:
                    data["merge"] = 0
                line = pd.DataFrame(data=data, index=[0])
                # print(line)
                # print(df)
                df = pd.concat([df.iloc[:idx], line, df.iloc[idx:]]).reset_index(drop=True)
            prev_offset = row.offset
    # print(df)

    first_row = df.iloc[0, :]
    last_row = df.iloc[-1, :]
    start_time = first_row['onset']
    end_time = last_row['offset']
    
    if start_time > task_onset:
        data = {'onset': task_onset, 'offset': start_time, 'toy': "no_ops"}
        for toy in toy_list:
            data[toy+"_ord"] = np.nan
        if merging:
            data["merge"] = 0
        line = pd.DataFrame(data=data, index = [0])
        df = pd.concat([line, df]).reset_index(drop=True)
    
    if end_time < task_offset:
        data = {'onset': end_time,
                        'offset': task_offset, 'toy': "no_ops"}
        for toy in toy_list:
            data[toy+"_ord"] = np.nan
        if merging:
            data["merge"] = 0
        line = pd.DataFrame(data=data, index = [len(df)])
        df = pd.concat([df, line]).reset_index(drop=True)
    df = df.sort_values(by=['onset'])
    df = df.reset_index(drop=True)
    return df

def redo_columns(df, toys_list, threshold, non_compatible_toys_dict):
    # for idx, row in df.itertuples():
    # print(df)
    df['merge'] = 0
    for idx, row in enumerate(df.itertuples()):
        for t in toys_list:
            # if t == 'broom':
                # print('here')
            non_compatible_toys = non_compatible_toys_dict[t]
            non_compatible_toys_cols = ['all_toys_'+i+"_ord" for i in non_compatible_toys]
            toy_col = 'all_toys_'+t+"_ord"
            if not np.isnan(getattr(row, toy_col)):
                last_valid_idx = df.loc[:idx-1, toy_col].last_valid_index()
                if not last_valid_idx is None:
                    if last_valid_idx > 0 and last_valid_idx < row.Index:
                        # print(df.loc[last_valid_idx:idx, non_compatible_toys_cols].isnull().values.all())
                        # print(row.all_toys_onset - df.loc[last_valid_idx, 'all_toys_offset'] )
                        if row.all_toys_onset - df.loc[last_valid_idx, 'all_toys_offset'] <= threshold and df.loc[last_valid_idx:idx, non_compatible_toys_cols].isnull().values.all():
                        # print(all_prev_val)
                            df.loc[last_valid_idx:idx-1, toy_col] = getattr(row, toy_col)
                            df.loc[idx, 'merge'] = 1
    return df

def pad_no_ops(df, task_onset, task_offset, merging = True):
    prev_offset = df['all_toys_offset'][0]
    # new_col = []
    for idx, row in enumerate(df.itertuples()):
    # print(row)
        if idx > 0 and idx < len(df):
            if row.all_toys_onset != prev_offset and row.all_toys_offset != task_offset:
                data = {'all_toys_onset': prev_offset, 'all_toys_offset': row.all_toys_onset}
                for col in df.columns:
                    if col != 'all_toys_onset' and col != 'all_toys_offset':
                        data[col] = np.nan
                line = pd.DataFrame(data=data, index  = [0])
                # print(data)
                df = pd.concat([df.iloc[:idx], line, df.iloc[idx:]]).reset_index(drop=True)
            prev_offset = row.all_toys_offset
    df = df.sort_values(by=['all_toys_onset'])
    cols = [col for col in each_task_df.columns if 'all_toys' in col]
    # print(df[cols+['all_toys_onset', 'all_toys_offset']])

    df.dropna(subset = ["all_toys_onset", "all_toys_offset",], inplace=True)
    df = df.reset_index(drop=True)    
    df = df.sort_values(by=['all_toys_onset'])
    return df

if __name__ == '__main__':
    # pre-defined vars
    CHECKING = False # if True will export the CSV and draw out the interaction timeline

    # clean_data_dict = {'MPS':{}, "MPM":{}, 'NMS':{}, "NMM":{}}
    clean_data_dict = {}
    floor_time = {}
    merge_stat = {}
    no_ops_merge_stat = {}
    merged_df_dict = {}
    original_df_dict = {'MPS': {}, 'MPM': {}, 'NMS': {}, 'NMM': {}}
    for no_ops_threshold in [5, 7, 10]:
        clean_data_dict[no_ops_threshold] = {'MPS': {}, 'MPM': {}, 'NMS': {}, 'NMM': {}}
        merge_stat[no_ops_threshold] = {'MPS': {}, 'MPM': {}, 'NMS': {}, 'NMM': {}}
        no_ops_merge_stat[no_ops_threshold] = {'MPS': {}, 'MPM': {}, 'NMS': {}, 'NMM': {}}
        merged_df_dict[no_ops_threshold] = {'MPS': {}, 'MPM': {}, 'NMS': {}, 'NMM': {}}


        # floor_time[no_ops_threshold] = {}
    # task_to_storing_dict = {'MPS': mps_dict,
    #                     'MPM': mpm_dict, 'NMS': nms_dict, 'NMM': nmm_dict}
    
    toy_names = {str(v): k for k, v in toys_dict.items()}
    data_path = './data/raw/20210824_output/*.csv'

    for file_name in glob.glob(data_path):
        print(file_name)
        subj = int(file_name.split('/')[-1].split('.')[0])
        # if subj == 8:
            # print(subj)
            # if subj != 39 and subj != 4 or (subj != 43 and task != 'NMM'):
        if subj == 1:
            print('here')
        df = pd.read_csv(file_name)
        # 20210721: floor_time indicates the parts of the sessions that were in included in Justine's analysis
        floor_time_onset = df['floortime.onset'].dropna().values
        floor_time_offset = df['floortime.offset'].dropna().values
        # print(floor_time_onset, floor_time_offset)
        df.columns = df.columns.str.replace(r'\s+', '')
        df.columns = df.columns.str.replace('.', '_')

        subj_task_onset_offset = {}
        floor_time[subj] = {}

        for task in tasks:
            if subj == 1 and task == 'MPM':
                print(floor_time_onset, floor_time_offset)
            floor_time[subj][task] = {}

            task_onset_offset = df.loc[df.loc[:, 'task_task'] == task, ['task_onset', 'task_offset']].to_numpy().flatten()
            task_onset, task_offset = task_onset_offset[0], task_onset_offset[1]
            
            subj_task_onset_offset[task] = [task_onset, task_offset]

            current_task_floor_time = []
            for idx, f_time_onset in enumerate(floor_time_onset):
                if f_time_onset >= task_onset and floor_time_offset[idx] <= task_offset:
                    current_task_floor_time.append([f_time_onset, floor_time_offset[idx]])
            floor_time[subj][task] = current_task_floor_time
                
            
            df_ = pd.DataFrame()
            df_list = []
            task_df_list = {}
            task_df = {}
            for no_ops_threshold in [5, 7, 10]:
                task_df_list[no_ops_threshold] = []
                task_df[no_ops_threshold] = pd.DataFrame()

            for idx, f_time_onset in enumerate(floor_time_onset):
                if f_time_onset >= task_onset and floor_time_offset[idx] <= task_offset:
                    f_time_offset = floor_time_offset[idx]
                   

                    each_task_df = df.loc[(df.loc[:,  'all_toys_onset'] >= f_time_onset) & (df.loc[:,  'all_toys_offset'] <= floor_time_offset[idx]), :].reset_index()
                    each_task_df = each_task_df.drop(columns='all_toys_ordinal')
                    cols = [col for col in each_task_df.columns if 'all_toys' in col]
                    # print(cols)

                    
                    original_df = each_task_df.copy()
                    to_merge_df = each_task_df.copy()

                    original_df['toy'] = original_df.apply(add_toy_to_single_col, axis=1, args=(toys_list,))
                    # print(original_df)


                    original_df = insert_no_ops(original_df[cols+['toy']], 'all_toys_offset', toys_list, f_time_onset, f_time_offset)
                    df_ = pd.concat([df_, original_df])
                    df_list.append(original_df)
                    # print(to_merge_df)
                    to_merge_df = pad_no_ops(to_merge_df, f_time_onset, f_time_offset)
                    # print(to_merge_df)

                        # print(each_task_df.columns)
                    for no_ops_threshold in [5, 7, 10]:
                        merged_df = redo_columns(to_merge_df.copy(), toy_to_task_dict[task], threshold = no_ops_threshold*1000, non_compatible_toys_dict= non_compatible_toys_dict)
                        # print(merged_df)
                       
                        merged_df['toy'] = merged_df.apply(add_toy_to_single_col, axis=1, args=(toys_list,))
                        # print(merged_df)
                        # merged_df.to_csv('./data/to_check/check_reading_data_20210805/check_redo_col/no_ops_threshold_'+str(no_ops_threshold)+'/'+str(subj)+'_'+task+"_"+str(idx)+'.csv', index = False)
                        merged_df = insert_no_ops(merged_df[cols+['toy', 'merge']], 'all_toys_offset', toys_list, f_time_onset, f_time_offset, merging = True)
                        # print(merged_df[cols+['toy']])

                        
                        # print(each_task_df.columns)
                        # print(df_.columns)

                        merged_df, no_ops_merge_cnt, no_ops_merge_time = remove_small_no_ops(merged_df, small_no_ops_threshold = no_ops_threshold*1000)
                        task_df_list[no_ops_threshold].append(merged_df)
                        task_df[no_ops_threshold] = pd.concat([task_df[no_ops_threshold], merged_df])
                        # merged_df, merge_cnt, merge_time = get_consecutive_toys(each_task_df, toys_of_interest_dict[task])
                        
                        # no_ops_merge_stat[no_ops_threshold] = (no_ops_merge_cnt, no_ops_merge_time)

                        # merge_stat[no_ops_threshold][task][subj] = (merge_cnt, merge_time)
                        # print(each_task_df)
                        # print(each_task_df)
                        # each_task_df = insert_no_ops(each_task_df[cols+['toy']], 'all_toys.offset', toys_list, f_time_onset, f_time_offset)

                        

                    # each_task_df = insert_no_ops(df_, 'offset', toys_list, task_onset, task_offset)
            df_ = df_.reset_index(drop = True)
            if len(df_) == 0:
                print('here')
            original_df_dict[task][subj] = df_list
            for no_ops_threshold in [5, 7, 10]:
                clean_data_dict[no_ops_threshold][task][subj] = task_df_list[no_ops_threshold]
                merged_df_dict[no_ops_threshold][task][subj] = task_df[no_ops_threshold]
            # clean_data_dict[task][subj] = df_list

            if CHECKING:
                path = Path('./figures/hmm/20210824/plain_timeline/original/'+task)
                path.mkdir(parents=True, exist_ok=True)
                plain_fig_name = './figures/hmm/20210824/plain_timeline/original/'+task+'/'+str(subj)+".png"
                # print(merged_df_dict[no_ops_threshold][task][subj])
                df_list = original_df_dict[task][subj]
                df_check = pd.DataFrame()

                for df_ in df_list:
                    df_check = pd.concat([df_check, df_])
                draw_plain_timeline(str(subj) + ', condition: ' + condition_name[task] +" threshold: " + str(no_ops_threshold), df_check, plain_fig_name)

                for no_ops_threshold in [5, 7, 10]:
                # plot out the timestep toy interaction to check
                # plain_fig_name = './figures/hmm/20210729/plain_timeline/start_from_container/'+task+'/'+str(subj)+".png"
                    path = Path('./figures/hmm/20210824/plain_timeline/no_ops_threshold_'+str(no_ops_threshold)+'/'+task)
                    path.mkdir(parents=True, exist_ok=True)
                    plain_fig_name = './figures/hmm/20210824/plain_timeline/no_ops_threshold_'+str(no_ops_threshold)+'/'+task+'/'+str(subj)+".png"
                    # print(merged_df_dict[no_ops_threshold][task][subj])
                    draw_plain_timeline(str(subj) + ', condition: ' + condition_name[task] +" threshold: " + str(no_ops_threshold), merged_df_dict[no_ops_threshold][task][subj], plain_fig_name)

    if CHECKING:
        subj_list = list(clean_data_dict[5]['MPS'].keys())
        for no_ops_threshold in [5, 7, 10]:
            path = Path('./data/to_check/check_reading_data_20210824/no_ops_threshold_'+str(no_ops_threshold))
            path.mkdir(parents=True, exist_ok=True)

        
        for subj in subj_list:
            for no_ops_threshold in [5, 7, 10]:
                df_check = pd.DataFrame()
                for task in tasks:
                    df_list = clean_data_dict[no_ops_threshold][task][subj]
                    for df_ in df_list:
                        df_check = pd.concat([df_check, df_])
                df_check = df_check[['onset', 'offset', 'toy']]
                
                df_check.to_csv('./data/to_check/check_reading_data_20210824/no_ops_threshold_'+str(no_ops_threshold)+'/'+str(subj)+'.csv', index = False)
    # # # print(clean_data_dict[no_ops_threshold])
    for no_ops_threshold in [5, 7, 10]:
        with open('./data/interim/20210824_'+str(no_ops_threshold)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'wb+') as f:
            pickle.dump(clean_data_dict[no_ops_threshold], f)
        with open('./data/interim/20210824_'+str(no_ops_threshold)+'_no_ops_threshold_toy_merge_stat.pickle', 'wb+') as f:
            pickle.dump(merge_stat, f)
        with open('./data/interim/20210824_'+str(no_ops_threshold)+'_no_ops_threshold_no_ops_merge_stat.pickle', 'wb+') as f:
            pickle.dump(no_ops_merge_stat, f)
    with open('./data/interim/20210824_floor_time.pickle', 'wb+') as f:
        pickle.dump(floor_time, f)
    with open('./data/interim/20210824_floor_time.pickle', 'wb+') as f:
        pickle.dump(floor_time, f)

    with open('./data/interim/20210824_clean_data_for_feature_engineering.pickle', 'wb+') as f:
        pickle.dump(original_df_dict, f)