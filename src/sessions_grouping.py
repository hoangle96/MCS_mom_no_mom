import numpy as np 
import pickle 
from variables import tasks
from pathlib import Path 

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from all_visualization_20210824 import rank_state
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
plt.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 20})

def get_state_frequency(state_seq, n_states):
    """
    Param: 
    state_seq: state_seq of a session 
    n_states: number of states of the model 
    """
    unique_state, cnt = np.unique(state_seq, return_counts = True)
    cnt = cnt/cnt.sum()

    return [cnt[unique_state==i].item() if i in unique_state else 0 for i in range(n_states)]

def draw_pca(pca_array, focus_explore_diff, diff_max, colormap, title, fig_path):
    fig = plt.figure(facecolor='white', figsize = (10, 10))
    color = np.array(focus_explore_diff)/diff_max
    plt.scatter(pca_array[:,0], pca_array[:,1], c = color, cmap = colormap)
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(top = 2.5, bottom = -3)
    plt.xlim(right = 3, left = -3)
    plt.title(title, fontsize = 18)
    plt.tight_layout()
    plt.savefig(fig_path, dpi = 300, facecolor=fig.get_facecolor(), transparent=True)       
    plt.close()


def draw_k_means(pca_array, focus_explore_diff, diff_max, colormap, marker, title, fig_path):
    fig = plt.figure(facecolor='white', figsize = (10, 10))
    color = np.array(focus_explore_diff)/diff_max
    color_list = [colormap(c) for c in color]
    # print(color)
    for i in range(len(pca_array)):
        # print(color[i])
        plt.scatter(pca_array[i,0], pca_array[i,1], color = color_list[i], marker = marker[i])
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    # for idx, center in enumerate(centers):
    #     plt.scatter(center[0], center[1], marker = marker_dict[idx], c = 'black')
    #     plt.annotate("center " + str(idx), (center[0], center[1]))

        # plt.scatter(center[0], center[1])

    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=14) 
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(top = 3, bottom = -3)
    plt.xlim(right = 3.5, left = -3)
    plt.title(title, fontsize = 18)
    plt.tight_layout()
    plt.savefig(fig_path, dpi = 300, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

def draw_tsne(pca_array, color_list, title, fig_path):
    fig = plt.figure(facecolor='white', figsize = (10, 10))
    # color = np.array(focus_explore_diff)/diff_max
    # color_list = [colormap(c) for c in color]
    # print(color)
    # for i in range(len(pca_array)):
        # print(color[i])
    plt.scatter(pca_array[:,0], pca_array[:,1], color = color_list)
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.ylim(top = 3, bottom = -3)
    # plt.xlim(right = 3.5, left = -3)
    plt.title(title, fontsize = 18)
    plt.tight_layout()
    plt.savefig(fig_path, dpi = 300, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

if __name__ == "__main__":
    fig_name_by_task = {
                        'MPS': "With caregivers, fine motor toys",
                        'MPM': "With caregivers, gross motor toys",
                        'NMS': "Without caregivers, fine motor toys",
                        'NMM': "Without caregivers, gross motor toys",
                    }
    feature_set = 'n_new_toy_ratio'
    interval_length = 1.5
    n_states = 5
    no_ops_time = 10
    model_file_name = "model_20210824_"+feature_set+"_"+str(interval_length)+"_interval_length_"+str(no_ops_time)+"_no_ops_threshold_"+str(n_states)+'_states.pickle'
    model_file_path = Path('./models/hmm/20210824__30s_offset/'+feature_set)/model_file_name
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    state_name_dict = rank_state(model)

    with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_prediction_'+str(interval_length)+'_min.pickle', 'rb') as f:
        merged_pred_dict_all = pickle.load(f)

    # with open():

    min_ = np.inf
    max_ = -np.inf
    all_task_freq_dict = {}
    focus_explore_diff_dict = {}
    x_new_dict = {}

    all_data = []
    pca = PCA(n_components=2)

    for task in tasks:
        # print(merged_pred_dict_all[task].keys())
        all_state_freq_this_task = []
        focus_explore_diff = []

        for infant_id, state_seq in merged_pred_dict_all[task].items():
            state_ratio = get_state_frequency(state_seq, n_states)
            sum_focus = state_ratio[1] + state_ratio[2]
            sum_explore = state_ratio[3]+ state_ratio[4]

            f_e_ratio_ = sum_focus - sum_explore  
            focus_explore_diff.append(f_e_ratio_)
            all_data.append(state_ratio)
            all_state_freq_this_task.append(state_ratio)
        if min_ > np.amin(focus_explore_diff):
            min_ = np.amin(focus_explore_diff)
        if max_ < np.amax(focus_explore_diff):
            max_ = np.amax(focus_explore_diff)
        # x_new = pca.fit_transform(np.array(all_state_freq_this_task))

        # x_new_dict[task] = x_new
        all_task_freq_dict[task] = all_state_freq_this_task
        focus_explore_diff_dict[task] = focus_explore_diff
        
    
    all_data = np.array(all_data).reshape((-1,5))
    mean_ = np.mean(all_data, axis = 0)
    std_ = np.std(all_data, axis = 0)
    x = (all_data-mean_)/std_

    pca.fit(x)
    print(pca.explained_variance_ratio_)
    norm = Normalize(vmin=min_, vmax=max_)
    colormap = cm.get_cmap('coolwarm')

    for task in tasks:
        x_original = np.array(all_task_freq_dict[task].copy())
        x_original = (x_original-mean_)/std_
        x_new = pca.transform(np.array(x_original))

        title = "PCA sessions, " + fig_name_by_task[task]
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"pca_"+task+".png"
        draw_pca(x_new, focus_explore_diff_dict[task], max_, colormap, title, fig_path)
    
    # do k-mean on the thing
    for k in [2,3,4]:
        marker_dict = {0: 'x', 1:'^', 2:"s", 3:"o", 4:"1"}
        color_dict = {0:'salmon', 1:"blue", 2: "violet", 3:"brown", 4:"green"}
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_data)
        centers = (kmeans.cluster_centers_ - mean_)/std_ 
        for task in tasks:
            marker = kmeans.predict(all_task_freq_dict[task])
            marker_list = [marker_dict[i] for i in marker]

            x_original = np.array(all_task_freq_dict[task].copy())
            x_original = (x_original-mean_)/std_
            x_new = pca.transform(x_original)

            title = fig_name_by_task[task]
            # print(color_list)
            # print(len(all_task_freq_dict[task]))
            # print(len(color_list))

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_"+task+"_"+str(k)+"_no_center.png"
            draw_k_means(x_new, focus_explore_diff_dict[task], max_, colormap, marker_list, title, fig_path)

        # tsne = TSNE(n_components=k, perplexity = 15).fit(all_data)
        for perplexity in [5, 10, 15, 20, 25]:
            for task in tasks:
                colors = kmeans.predict(all_task_freq_dict[task])
                color_list = [color_dict[i] for i in colors]

                x_original = np.array(all_task_freq_dict[task].copy())
                # x_original = (x_original-mean_)/std_
                x_new = TSNE(n_components=2, perplexity = perplexity).fit_transform(x_original)

                title = fig_name_by_task[task]
                # print(color_list)
                # print(len(all_task_freq_dict[task]))
                # print(len(color_list))

                fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+task+"_"+str(k)+"_perp_"+str(perplexity)+".png"
                draw_tsne(x_new, color_list, title, fig_path)

