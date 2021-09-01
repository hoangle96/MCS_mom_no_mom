#%%
import numpy as np
import pomegranate as pom
import pickle
from variables import tasks
# %%
# feature_set = 
# interval_length = 2
# model_file_name = "model_20210815_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
# model_file_path = Path('./models/hmm/20210815/'+feature_set)/model_file_name
# with open(model_file_path, 'wb+') as f:
#     pickle.dump(model, f)

# %%
feature_set = 'n_new_toy_ratio'
no_ops_time = 7
interval_length = 1.5
with open("../data/interim/20210815_"+str(no_ops_time)+"_no_ops_threshold_feature_dict_with_"+feature_set+"_"+str(interval_length)+"_min.pickle", 'rb') as f:
    feature_dict = pickle.load(f)
# %%
feature_dict["MPS"][1][0.0]

# %%
n_features = 4
input_list = np.empty((0, n_features))

for task in tasks:
    for subj, shifted_df_dict in feature_dict[task].items():
        for shift_time, feature_vector in shifted_df_dict.items():
            # if feature_set == 'n_new_toy_ratio_and_fav_toy_till_now':
                # m, n, _ = feature_vector.shape
                # feature_vector = feature_vector.reshape((n, m))
            # print(feature_vector.shape)

            input_list = np.vstack((input_list, feature_vector))
            # len_list.append(len(feature_vector))
# %%
np.unique(input_list[:,2])
new_toys_bin = [.2, .4, .6, .8, 1]
discretized_n_new_toys = np.digitize(input_list[:,2].copy(), new_toys_bin, right = False)
np.unique(discretized_n_new_toys)
# %%
fav_toy_bin = [0, .2, .4, .6, .8]
fav_toy_rate_discretized = np.digitize(input_list[:,3].copy(), fav_toy_bin, right = False)
np.unique(fav_toy_rate_discretized)

# %%
