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