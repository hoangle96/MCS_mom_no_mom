def add_toy_to_single_column(row, toy_list):
    val = []

    for toy in toy_list:
        if row[toy] == 1:
            val.append(toy)
    
    if len(val) == 0:
        val = ["no_ops"]
    return val

def convert_to_one_action(toy_df, toy_cols):
    keep_cols = [cols for cols in toy_cols if 'ordinal' in cols or 'onset' in cols or 'offset' in cols]
    action_cols = [cols for cols in toy_cols if 'ordinal' not in cols and 'onset' not in cols and 'offset' not in cols]
    toy_df[action_cols] = toy_df[action_cols].replace({'y':1,'n':0})
    temp = toy_df[action_cols[0]]
    for col in action_cols[1:]:
        temp = temp | toy_df[col]
    toy_df['interaction'] = temp

    return toy_df[keep_cols + ['interaction']]


def contain(onset, offset, toy_df, toy_name):
    if len(toy_df) > 0:
        onset_col = toy_name + ".onset"
        offset_col =  toy_name + ".offset"
        # print(onset, offset)
        # print(toy_df)
        to_return = toy_df.loc[((toy_df.loc[:, onset_col] >= onset) & (toy_df.loc[:, offset_col] <= offset))\
                    | ((toy_df.loc[:, onset_col] >= onset) & (toy_df.loc[:, onset_col] <= offset))\
                    | ((toy_df.loc[:, offset_col] >= onset) & (toy_df.loc[:, offset_col] <= offset)), 'interaction'].to_numpy()
    else:
        to_return = ()
    return to_return


def create_common_df(onset, offset, toys_list, toy_df_list):
    # based on the ruby script
    onsets = [onset]
    offsets = [offset]

    for toy_idx, toy in enumerate(toys_list):
        onsets.extend(toy_df_list[toy_idx][toy+'.onset'])
        offsets.extend(toy_df_list[toy_idx][toy+'.offset'])

    times = sorted(np.unique(onsets+ offsets))

    onset_list = []
    offset_list = []

    onset_time = times[0]
    toy_interaction_list = [[] for _ in range(len(toys_list))]

    if len(times) > 0:
        for offset_time in times[1:]:
            # offset_list.append(onset_time)
            for toy_idx, toy in enumerate(toys_list):
                within_boundary = contain(onset_time, offset_time, toy_df_list[toy_idx], toy)
                if len(within_boundary) > 0 :
                    interaction = within_boundary
                    toy_interaction_list[toy_idx].append(interaction[0])
                else:
                    # print(toy_interaction_list[toy_idx])
                    toy_interaction_list[toy_idx].append(0)
            onset_list.append(onset_time)
            onset_time = offset_time
            offset_list.append(offset_time)
    data_dict = {t: toy_interaction_list[t_idx] for t_idx, t in enumerate(toys_list)}
    data_dict['onset']  = onset_list
    data_dict['offset'] = offset_list
    return pd.DataFrame(data=data_dict)


def get_feature_every_interval(df: pd.DataFrame, interval_length: int, no_ops_threshold: int):
    """
    For every 'interval_length', get the features set (# toys changes per min, # toys per min, # new toys, fav toy ratio, toy IOU) 

    Parameters:
    -----------
        df: dataframe that includes the interaction for one session. Must have 3 columns: 'onset' (start of interaction), 'offset' (end of interaction), and 'toy' (toys are being played, 'no_ops' if no toys are interacted)
        interaval_length: 
        no_ops_threshold: shorter 'no_ops' episode might be due to accidental drop, as such, we eliminate 'no_ops' occurance that is shorter than 'no_ops_threshold' 

    Returns:
    --------
        list of features
            time_arr: time for ever
            idx_arr, 
            ep_rate_list, 
            toy_per_min_list, 
            toy_per_sc_list
    """
    window_time = interval_length*60000
    first_row = df.iloc[0, :]
    last_row = df.iloc[-1, :]
    start_time = first_row['onset']
    end_time = last_row['offset']
    ptr = start_time + window_time
    left_bound = start_time

    ep_rate_list = []
    toy_per_min_list = []
    toy_per_sc_list = []

    time_arr = []
    idx_arr = []

    while ptr < end_time:
        if ptr > end_time:
            ptr = end_time
        if ptr - left_bound < window_time*1/3:
            break

        idx = find_neighbours(df, ptr)
        if idx not in idx_arr:

            ep_rate, toy_rate, toy_per_sc = get_feature_vector(
                left_bound, ptr, df, no_ops_threshold)
            ep_rate_list.append(ep_rate)
            toy_per_min_list.append(toy_rate)
            toy_per_sc_list.append(toy_per_sc)
            time_arr.append(ptr)
            idx_arr.append(idx)

        left_bound = ptr
        ptr += window_time

    return time_arr, idx_arr, ep_rate_list, toy_per_min_list, toy_per_sc_list

def find_neighbours(df, value):
    exactmatch = df[df['timeframe'] == value]
    if len(exactmatch) != 0:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df['timeframe'] < value].timeframe.idxmax()
        # upperneighbour_ind = df[df['timeframe']>value].timeframe.idxmin()
        return lowerneighbour_ind  # , upperneighbour_ind]


def get_feature_vector(left_pt, right_pt, df, no_ops_threshold):
    curr = df.loc[(df.loc[:, 'timeframe'] <= right_pt) &
                  (df.loc[:, 'timeframe'] >= left_pt), :]
    curr_ = curr.loc[curr.loc[:, 'last_row'] == 0, :]
    small_no_ops = curr_.loc[(curr.loc[:, 'toy'] == 'no_ops') & (
        curr.loc[:, 'duration'] < no_ops_threshold), :]

    segment_len = (right_pt - left_pt)/60000
    n_episode = len(curr_) - len(small_no_ops)

    toy_list = curr_.loc[:, 'toy']

    toy_list_flatten = []
    for toy in toy_list:
        if isinstance(toy, tuple):
            for current_t in toy:
                toy_list_flatten.append(current_t)
        else:
            toy_list_flatten.append(toy)
    toy_list_flatten_final = [t_ for t_ in toy_list_flatten if t_ != 'no_ops']
    unique_toys = np.unique(toy_list_flatten_final)

    segment_len = (right_pt - left_pt)/60000

    if n_episode == 0:
        toy_per_sc = len(unique_toys)
    else:
        toy_per_sc = len(unique_toys)/n_episode
    return n_episode/segment_len, len(unique_toys)/segment_len, toy_per_sc

def get_consecutive_toys(df, toys_of_interest):
    merge_time = 0
    merge_cnt = 0
    df = df.reset_index(drop=True)
    df['merged_toy'] = 0
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
            contained_tuple = tuple(contained)
            container_tuple = tuple([container])
        else:
            proceed = df[container+"_ord"].notna().any() and df[contained+"_ord"].notna().any()
            toys_of_interest_list = [container, contained]
            contained_tuple = tuple([contained])
            container_tuple = tuple([container])


        if proceed:
            df = df.reset_index()
            # print(df)
            df['duration'] = df['offset']  - df['onset'] 
            df['contain'] = 0

            df['contained_toys_only'] = 0
            df['container_toys_only'] = 0
            df['container_toys'] = 0
            # check to see if the row contains the toys we care about
            df_toy_np = df.toy.tolist()
            # condition: the toys in the row are a subset of subset of the toys of interest
            toy_of_interest_set = set(toys_of_interest_list)
            row_condition = [set(i).issubset(toy_of_interest_set) for i in df_toy_np]
            df.loc[row_condition, 'contain'] = 1

            contained_row_condition = [set(i).issubset(contained_tuple) for i in df_toy_np]
            df.loc[contained_row_condition, 'contained_toys_only'] = 1

            container_row_condition = [set(i).issubset(container_tuple) for i in df_toy_np]
            df.loc[container_row_condition, 'container_toys_only'] = 1

            container_row_condition = [set(container_tuple).issubset(set(i)) for i in df_toy_np]
            df.loc[container_row_condition, 'container_toys'] = 1

            # print(df.loc[:,'toy'])
            # print(df)

            # if the infants play with the contained toys or container toys for a long time **needs some qualification here, set to 30s for now**
            # print(df.loc[(df.loc[:,'contain'] == 1) & (df.loc[:,'toy'].isin(contained_tuple))])
            df.loc[(df.loc[:,'contain'] == 1) & (df.loc[:,'contained_toys_only'] == 1) & (df.loc[:,'duration'] >= 12 * 1000), 'contain'] = 0
            df.loc[(df.loc[:,'contain'] == 1) & (df.loc[:,'container_toys_only'] == 1) & (df.loc[:,'duration'] >= 12 * 1000), 'contain'] = 0
            
            # print(df)
            # group the consecutive rows that have the toy we care together
            df['g'] = df['contain'].ne(df['contain'].shift()).cumsum()
            # print(df[['toy', 'contain', 'contained_toys_only', 'container_toys_only', 'container_toys', 'g']])

            for group in df['g'].unique():
                if len(df.loc[(df['g'] == group)&(df['contain'] == 1), :]) > 1:
                    merged = False
                    toy_playing_now = df.loc[(df['g'] == group)&(df['contain'] == 1), 'toy'].to_numpy().tolist()
                    toy_playing_now = tuple(set(itertools.chain.from_iterable(toy_playing_now)))
                    # cases where there are three toys, check to make sure not over-merging
                    if len(toy_playing_now) == 3:
                        toy_set = df.loc[(df['g'] == group)&(df['contain'] == 1), 'toy'].to_numpy().tolist()
                        current_df = df.loc[(df['g'] == group)&(df['contain'] == 1), :].copy()
                        current_df['check_three_toys'] = 0
                        for idx, t in enumerate(contained, 1):
                            contained_row_condition = [set(i).issubset([container, t]) for i in toy_set]
                            current_df.loc[contained_row_condition & (current_df.loc[:,'check_three_toys'] == 0), 'check_three_toys'] = idx
                            # print(current_df)
                        contained_row_condition = [toy_of_interest_set.issubset(set(i)) for i in toy_set]
                        # current_df.loc[contained_row_condition, 'check_three_toys'] = 3

                        # if consecutive and long duration -> don't merge
                        current_df['check_g'] = current_df['check_three_toys'].ne(current_df['check_three_toys'].shift()).cumsum()
                        # for check_group in current_df['check_g'].unique():
                        print(current_df[["onset", "offset", "check_three_toys", "check_g", "toy"]])
                        sub_df = current_df.loc[(current_df['check_g'] == 1)]
                        first_idx = sub_df.index.tolist()[0]
                        last_idx = sub_df.index.tolist()[-1]
                        print(sub_df[["onset", "offset", "check_three_toys", "check_g", "toy"]])

                        check_duration = sub_df.loc[last_idx, 'offset'] - sub_df.loc[first_idx, 'onset']
                        if check_duration > 12*1000:
                            # merge the first one
                            curr_toy_set = sub_df['toy'].to_numpy().tolist()
                            curr_toy_set = tuple(set(itertools.chain.from_iterable(curr_toy_set)))

                            df.iloc[first_idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
                            df.at[first_idx, 'toy'] = curr_toy_set
                            df.at[first_idx,'merged_toy'] = 1
                            merge_cnt += 1
                            merge_time += df.iloc[last_idx, df.columns.get_loc('offset')] - df.iloc[first_idx, df.columns.get_loc('onset')]
                            df = df.drop(index=range(first_idx+1, last_idx+1))
                            df = df.reset_index(drop=True)

                            # sub_df = current_df.loc[(current_df['check_g'] > 1)]

                            # first_idx = sub_df.index.tolist()[0]
                            # last_idx = sub_df.index.tolist()[-1]
                            group_list = current_df['check_g'].unique()
                            for g_ in group_list[1:]:
                                df_include = current_df.loc[(current_df['contain'] ==1)&(current_df.loc[:,'container_toys'] == 1)&(current_df['check_g'] == g_), :]
                                if len(df_include) > 0:
                                    print(df_include[["onset", "offset", "check_three_toys", "check_g", "toy"]])
                                    first_idx = df_include.index.tolist()[0]
                                    last_idx = df_include.index.tolist()[-1]
                                    df.iloc[first_idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
                                    df.at[first_idx, 'toy'] = curr_toy_set
                                    df.at[first_idx,'merged_toy'] = 1
                                    merge_cnt += 1
                                    merge_time += df.iloc[last_idx, df.columns.get_loc('offset')] - df.iloc[first_idx, df.columns.get_loc('onset')]
                                    df = df.drop(index=range(first_idx+1, last_idx+1))
                                    df = df.reset_index(drop=True)
                            merged = True
                        else:
                            df_include = df.loc[(df['g'] == group)&(df['contain'] >= 1)&(df.loc[:,'container_toys'] == 1), :]
                            if len(df_include) > 0:
                                idx = df_include.index.tolist()[0]
                                df_ = df.loc[(df['g'] == group)&(df['contain'] == 1), :]
                                print(df_[["onset", "offset", "check_three_toys", "check_g", "toy"]])
                                last_idx = df_.index.tolist()[-1]
                                df.iloc[idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
                                df.at[idx, 'toy'] = toy_playing_now
                                df.at[idx,'merged_toy'] = 1
                                merge_cnt += 1
                                merge_time += df.iloc[last_idx, df.columns.get_loc('offset')] - df.iloc[idx, df.columns.get_loc('onset')]
                                df = df.drop(index=range(idx+1, last_idx+1))
                                df = df.reset_index(drop=True)
                    else: # if not merged:
                    # start with the container in the toy sets
                        df_include = df.loc[(df['g'] == group)&(df['contain'] >= 1)&(df.loc[:,'container_toys'] == 1), :]
                        # print(df_include[['toy', 'contain', 'contained_toys_only', 'container_toys_only', 'container_toys', 'g']])
                        if len(df_include) > 0:
                            idx = df_include.index.tolist()[0]
                            df_ = df.loc[(df['g'] == group)&(df['contain'] == 1), :]
                            print(df_)
                            last_idx = df_.index.tolist()[-1]
                            df.iloc[idx, df.columns.get_loc('offset')] = df.iloc[last_idx, df.columns.get_loc('offset')]
                            df.at[idx, 'toy'] = toy_playing_now
                            df.at[idx,'merged_toy'] = 1
                            merge_cnt += 1
                            merge_time += df.iloc[last_idx, df.columns.get_loc('offset')] - df.iloc[idx, df.columns.get_loc('onset')]
                            df = df.drop(index=range(idx+1, last_idx+1))
                            df = df.reset_index(drop=True)
    df['duration'] = df['offset'] - df['onset']
    # print(df)
    return df, merge_cnt, merge_time
