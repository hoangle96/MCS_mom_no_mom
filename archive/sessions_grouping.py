import numpy as np 
import pickle 
from variables import tasks, toy_to_task_dict
from pathlib import Path 

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram
from scipy import stats
from all_visualization_20210824 import rank_state
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
import matplotlib.lines as mlines
plt.style.use('seaborn')
matplotlib.rcParams.update({'font.size': 20})
from feature_engineering import rank_toy_local, get_first_last_time
import pandas as pd
plt.style.use('seaborn')
# from scipy.cluster import hierarchy

def plot_dendrogram(model,labels, colors, label_colors, title, fig_path, **kwargs):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html 
    # Create linkage matrix and then plot the dendrogram
    plt.figure(figsize = (20,10))
    plt.title('Hierarchical Clustering Dendrogram')
    handles = []
    for task, c in zip(labels, colors):
        handles.append(mlines.Line2D([], [], marker = "o", linestyle='None', mfc = c, markersize=10, fillstyle = 'full', label=task))
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    print(model.children_.shape)
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, show_leaf_counts = False, get_leaves = True, **kwargs)

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(label_colors[lbl.get_text()])
    plt.xlabel("Index of point, colored by the condition",fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 8)
    plt.legend(handles = handles, title="data point-leaf node")
    plt.title(title, fontsize = 24)
    plt.savefig(fig_path, dpi = 300)
    plt.close()

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
    plt.scatter(pca_array[:,0], pca_array[:,1], c = color, cmap = colormap, s = 50)
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22) 
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(top = 2.5, bottom = -3)
    plt.xlim(right = 3, left = -3)
    plt.title(title, fontsize = 22)
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
        plt.scatter(pca_array[i,0], pca_array[i,1], color = color_list[i], marker = marker[i], s = 80)
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    # for idx, center in enumerate(centers):
    #     plt.scatter(center[0], center[1], marker = marker_dict[idx], c = 'black')
    #     plt.annotate("center " + str(idx), (center[0], center[1]))

        # plt.scatter(center[0], center[1])

    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=14) 
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(top = 3, bottom = -3)
    plt.xlim(right = 3.5, left = -3)
    plt.title(title, fontsize = 22)
    plt.tight_layout()
    plt.savefig(fig_path, dpi = 300, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

def draw_kmeans_for_all(pca_array, color_list, marker_list, tasks, title, fig_path):
    fig = plt.figure(facecolor='white', figsize = (10, 10))
    for i in range(len(pca_array)):
        # print(color[i])
        plt.scatter(pca_array[i,0], pca_array[i,1], color = color_list[i], marker = marker_list[i], s = 50)
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # plt.ylim(top = 3, bottom = -3)
    # plt.xlim(right = 3.5, left = -3)
    plt.title(title, fontsize = 22)

    handles = []
    for idx, marker in enumerate(np.unique(marker_list)):
        handles.append(mlines.Line2D([], [], marker = marker, linestyle='None', markersize=10, label=tasks[idx]))
    
    plt.legend(handles = handles)
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
    plt.scatter(pca_array[:,0], pca_array[:,1], color = color_list, s = 50)
    plt.grid(False)
    plt.axhline(y = 0, color = 'black', linewidth = 0.5)
    plt.axvline(x = 0, color = 'black', linewidth = 0.5)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # plt.ylim(top = 3, bottom = -3)
    # plt.xlim(right = 3.5, left = -3)
    plt.title(title, fontsize = 22)
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

    with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_theshold_'+str(n_states)+'_states_toy_pred_dict_'+str(interval_length)+'_min.pickle', 'rb') as f:
        toy_pred_list = pickle.load(f)
                    
    with open('./data/interim/20210824'+feature_set+'_'+str(no_ops_time)+'_no_ops_threshold'+str(n_states)+'_states_merged_locomotion_'+str(interval_length)+'_min.pickle', 'rb') as f:
        merged_pred_w_locomotion = pickle.load(f)

    with open('./data/interim/20210824_'+str(no_ops_time)+'_no_ops_threshold_clean_data_for_feature_engineering.pickle', 'rb') as f:
        task_to_storing_dict = pickle.load(f)
    
    with open('./data/interim/20210824_floor_time.pickle', 'rb') as f:
        floor_time = pickle.load(f)

    # features: state-time feature
    all_data = []
    all_task_freq_dict = {}

    focus_explore_diff_dict = {}
    toy_play_time_dict = {}
    toy_play_time_vector = np.empty((0, 6))
    all_marker_list = []
    all_marker_mom = []
    all_marker_toy = []
    color_dict = {0:'salmon', 1:"blue", 2: "violet", 3:"brown", 4:"green"}
    marker_dict = {0: 'P', 1:'^', 2:"s", 3:"o", 4:"P"}
    min_ = np.inf
    max_ = -np.inf       
    # locomotion feature
    movement_feature = np.empty((0, n_states))
    movement_dict = {}
    for task_idx, task in enumerate(tasks):
        movement_dict[task] = np.empty((0, n_states))
        toy_play_time_dict[task] = np.empty((0, 6))
        for subj, df in merged_pred_w_locomotion[task].items():
            # get state time distribution
            df['pred'] = df['pred'].replace(state_name_dict)
            # print(df.head())

            df['duration'] = df['offset'] - df['onset']
            # print(df.groupby(['pred'])['movement_time'].sum())
            movement_per_state = (df.groupby(['pred'])['movement_time'].sum()/df.groupby(['pred'])['duration'].sum()).to_dict()
            movement_list = []
            for state in range(n_states):
                if not str(state) in movement_per_state.keys():
                    movement_list.append(0)
                else:
                    movement_list.append(movement_per_state[str(state)])
            movement_feature = np.vstack((movement_feature, np.array(movement_list).reshape((1,5))))
            movement_dict[task] = np.vstack((movement_dict[task], np.array(movement_list).reshape((1,5))))


            # get toy distribution, remove toy id by ranking
            sess_len = 0
            for f_time in floor_time[subj][task]:
                sess_len += f_time[-1]-f_time[0]

            subj_df = pd.DataFrame()
            for df_ in task_to_storing_dict[task][subj]:
                subj_df = pd.concat([subj_df, df_])
            start_time, end_time = get_first_last_time(subj_df)
            toy_rank = rank_toy_local(subj_df, toy_to_task_dict[task], start_time, end_time)
            toy_play_time = np.array(list(toy_rank.values())).reshape((1, -1))/sess_len
            toy_play_time = toy_play_time if task in ['MPS', 'NMS'] else toy_play_time[:,1:]
            toy_play_time_vector = np.vstack((toy_play_time_vector, toy_play_time))
            toy_play_time_dict[task] = np.vstack((toy_play_time_dict[task], toy_play_time))

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
        all_task_freq_dict[task] = all_state_freq_this_task
        focus_explore_diff_dict[task] = focus_explore_diff


        if task in ["MPS", 'MPM']:
            all_marker_mom.extend([marker_dict[0]]*len(movement_dict[task]))
        elif task in ["NMS", 'NMM']:
            all_marker_mom.extend([marker_dict[1]]*len(movement_dict[task]))

        if task in ["MPS", 'NMS']:
            all_marker_toy.extend([marker_dict[0]]*len(movement_dict[task]))
        elif task in ["MPM", 'NMM']:
            all_marker_toy.extend([marker_dict[1]]*len(movement_dict[task]))
        all_marker_list.extend([marker_dict[task_idx]]*len(movement_dict[task]))

    # for arr in movement_feature.T: #do not need the loop at this point, but looks prettier
    #     print(stats.describe(arr))
    x_movement_feature = (movement_feature - np.mean(movement_feature, axis = 0))/np.std(movement_feature, axis = 0)

    state_locomotion_time = np.hstack((all_data, movement_feature))
    state_toy_playtime = np.hstack((all_data, toy_play_time_vector))
    state_toy_locomotion = np.hstack((all_data, movement_feature, toy_play_time_vector))
    
    
    mean_state_toy_locomotion_ = np.mean(state_toy_locomotion, axis = 0) 
    std_state_toy_locomotion_ = np.std(state_toy_locomotion, axis = 0) 

    mean_locomotion_state_ = np.mean(state_locomotion_time, axis = 0) 
    std_locomotion_state_ = np.std(state_locomotion_time, axis = 0) 

    mean_toy_playtime_state_ = np.mean(state_toy_playtime, axis = 0) 
    std_toy_playtime_state_ = np.std(state_toy_playtime, axis = 0) 

    mean_ = np.mean(all_data, axis = 0) 
    std_ = np.std(all_data, axis = 0) 

    x_state_movement = (state_locomotion_time-mean_locomotion_state_)/std_locomotion_state_
    state_toy_normed = (state_toy_playtime-mean_toy_playtime_state_)/std_toy_playtime_state_
    state_toy_locomotion_normed = (state_toy_locomotion - mean_state_toy_locomotion_)/std_state_toy_locomotion_

    all_data = np.array(all_data).reshape((-1,5))

    mom_label_colors = {}
    toy_label_colors = {}
    for i in range(160):
        if i // 40 < 2:
            mom_label_colors[str(i)] = 'r'
        elif i //40 >=2:
            mom_label_colors[str(i)] = 'b'
        
        if (i // 40) % 2 == 0:
            toy_label_colors[str(i)] = 'r'
        elif (i // 40) % 2 == 1:
            toy_label_colors[str(i)] = 'b'

    ### Hierical clustering
    model = AgglomerativeClustering(distance_threshold=1, n_clusters=None, compute_distances = True)
    model = model.fit(x_state_movement)
    fig_path = "./figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"dendogram_hierarchical_cluster_state_movement_by_mom_normalized.png"
    plot_dendrogram(model, ["With CG", "Without CG"], ['r', 'b'], mom_label_colors, "Hierarchial graph, using both states and portion of time moving", fig_path)
    
    fig_path = "./figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"dendogram_hierarchical_cluster_state_movement_by_toy_normalized.png"
    plot_dendrogram(model, ["Fine-motor toys", "Gross-motor toys"], ['r', 'b'], toy_label_colors, "Hierarchial graph, using both states and portion of time moving", fig_path)
    
    # with state-toy
    model = AgglomerativeClustering(distance_threshold=1, n_clusters=None, compute_distances = True)
    model = model.fit(state_toy_normed)
    fig_path = "./figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"dendogram_hierarchical_cluster_state_toy_by_mom_normalized.png"
    plot_dendrogram(model, ["With CG", "Without CG"], ['r', 'b'], mom_label_colors, "Hierarchial graph, using both states and portion of time moving", fig_path)
    
    fig_path = "./figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"dendogram_hierarchical_cluster_state_toy_by_toy_normalized.png"
    plot_dendrogram(model, ["Fine-motor toys", "Gross-motor toys"], ['r', 'b'], toy_label_colors, "Hierarchial graph, using both states and portion of time moving", fig_path)
    
    # with all
    model = AgglomerativeClustering(distance_threshold=1, n_clusters=None, compute_distances = True)
    model = model.fit(state_toy_locomotion_normed)
    fig_path = "./figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"dendogram_hierarchical_cluster_state_toy_locomotion_by_mom_normalized.png"
    plot_dendrogram(model, ["With CG", "Without CG"], ['r', 'b'], mom_label_colors, "Hierarchial graph, using both states and portion of time moving", fig_path)
    
    fig_path = "./figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"dendogram_hierarchical_cluster_state_toy_locomotion_by_toy_normalized.png"
    plot_dendrogram(model, ["Fine-motor toys", "Gross-motor toys"], ['r', 'b'], toy_label_colors, "Hierarchial graph, using both states and portion of time moving", fig_path)
    

   
    colormap = cm.get_cmap('coolwarm')

    for k in [2,3,4]:
        # with state-freq and movement          
        kmeans = KMeans(n_clusters=k, random_state=0).fit(state_locomotion_time)
        all_transformed_data = kmeans.predict(state_locomotion_time)
        all_colors_list = [color_dict[i] for i in all_transformed_data]
        pca = PCA(n_components=2)
        pca.fit(x_state_movement)
        all_pca_x = pca.transform(x_state_movement)

        title = "All conditions, using state and movement as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_"+str(k)+".png"
        draw_kmeans_for_all(all_pca_x, all_colors_list, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

        title = "All conditions, using state and movement as feature,  kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_"+str(k)+"_mom.png"
        draw_kmeans_for_all(all_pca_x, all_colors_list, all_marker_mom, ["With caregiver", "Without caregiver"], title, fig_path)

        title = "All conditions, using state and movement as feature,  kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_"+str(k)+"_toy.png"
        draw_kmeans_for_all(all_pca_x, all_colors_list, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"], title, fig_path)

        # state-toy feature
        kmeans = KMeans(n_clusters=k, random_state=0).fit(state_toy_normed)
        all_colors_state_toy = kmeans.predict(state_toy_normed)
        all_colors_list_state_toy = [color_dict[i] for i in all_colors_state_toy]
        pca = PCA(n_components=2)
        pca.fit(state_toy_normed)
        all_pca_x_state_toy = pca.transform(state_toy_normed)

        title = "All conditions, using state and toy as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_state_toy_"+str(k)+".png"
        draw_kmeans_for_all(all_pca_x_state_toy, all_colors_list_state_toy, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

        title = "All conditions, using state and toy as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_state_toy_"+str(k)+"_mom.png"
        draw_kmeans_for_all(all_pca_x_state_toy, all_colors_list_state_toy, all_marker_mom, ["With caregiver", "Without caregiver"], title, fig_path)

        title = "All conditions, using state and toy as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_state_toy_"+str(k)+"_toy.png"
        draw_kmeans_for_all(all_pca_x_state_toy, all_colors_list_state_toy, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"], title, fig_path)


        ## all features
        kmeans = KMeans(n_clusters=k, random_state=0).fit(state_toy_locomotion_normed)
        all_colors_state_toy = kmeans.predict(state_toy_locomotion_normed)
        all_colors_list_state_toy_locomotion = [color_dict[i] for i in all_colors_state_toy]
        pca = PCA(n_components=2)
        pca.fit(state_toy_locomotion_normed)
        all_pca_x_state_toy_locomotion = pca.transform(state_toy_locomotion_normed)

        title = "All conditions, using state, locomotion and toy as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_state_toy_locomotion"+str(k)+".png"
        draw_kmeans_for_all(all_pca_x_state_toy_locomotion, all_colors_list_state_toy_locomotion, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

        title = "All conditions, using state, locomotion and toy as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_state_toy_locomotion"+str(k)+"_mom.png"
        draw_kmeans_for_all(all_pca_x_state_toy_locomotion, all_colors_list_state_toy_locomotion, all_marker_mom, ["With caregiver", "Without caregiver"], title, fig_path)

        title = "All conditions, using state, locomotion and toy as feature, kmeans clustering, k = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_state_toy_locomotion"+str(k)+"_toy.png"
        draw_kmeans_for_all(all_pca_x_state_toy_locomotion, all_colors_list_state_toy_locomotion, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"], title, fig_path)

        # with state-freq and movement          
        spectral_clustering = SpectralClustering(n_clusters= k, assign_labels='discretize', random_state=0).fit(state_locomotion_time)
        predicted_ = spectral_clustering.labels_
        all_colors_spectral = [color_dict[i] for i in predicted_]

        title = "All conditions, spectral clustering, using toy and locomotion time as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_"+str(k)+".png"
        draw_kmeans_for_all(all_pca_x, all_colors_spectral, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

        title = "All conditions, spectral clustering, using toy and locomotion time as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_"+str(k)+"_mom.png"
        draw_kmeans_for_all(all_pca_x, all_colors_spectral, all_marker_mom,  ["With caregiver", "Without caregiver"], title, fig_path)

        title = "All conditions, spectral clustering, using toy and locomotion time as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_"+str(k)+"_toy.png"
        draw_kmeans_for_all(all_pca_x, all_colors_spectral, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"], title, fig_path)

        # state-toy feature
        spectral_clustering = SpectralClustering(n_clusters= k, assign_labels='discretize', random_state=0).fit(state_toy_normed)
        predicted_ = spectral_clustering.labels_
        all_colors_spectral = [color_dict[i] for i in predicted_]

        title = "All conditions, spectral clustering, using toy and state as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_toy_"+str(k)+".png"
        draw_kmeans_for_all(all_pca_x_state_toy, all_colors_spectral, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

        title = "All conditions, spectral clustering, using toy and state as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_toy_"+str(k)+"_mom.png"
        draw_kmeans_for_all(all_pca_x_state_toy, all_colors_spectral, all_marker_mom,  ["With caregiver", "Without caregiver"], title, fig_path)

        title = "All conditions, spectral clustering, using toy and state as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_toy_"+str(k)+"_toy.png"
        draw_kmeans_for_all(all_pca_x_state_toy, all_colors_spectral, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"], title, fig_path)

        # all features
        spectral_clustering = SpectralClustering(n_clusters= k, assign_labels='discretize', random_state=0).fit(state_toy_locomotion_normed)
        predicted_ = spectral_clustering.labels_
        all_colors_spectral = [color_dict[i] for i in predicted_]

        title = "All conditions, spectral clustering, using locomotion, toy and state as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_toy_locomotion_"+str(k)+".png"
        draw_kmeans_for_all(all_pca_x_state_toy_locomotion, all_colors_spectral, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

        title = "All conditions, spectral clustering, using locomotion, toy and state as features, n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_toy_locomotion_"+str(k)+"_mom.png"
        draw_kmeans_for_all(all_pca_x_state_toy_locomotion, all_colors_spectral, all_marker_mom,  ["With caregiver", "Without caregiver"], title, fig_path)

        title = "All conditions, spectral clustering, using locomotion, toy and state as features,n_clusters = " + str(k)
        fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"spectral_toy_locomotion_"+str(k)+"_toy.png"
        draw_kmeans_for_all(all_pca_x_state_toy_locomotion, all_colors_spectral, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"], title, fig_path)

        # for idx, task in enumerate(tasks):
        #     # with state-freq and movement          
        #     x_original = np.hstack((np.array(all_task_freq_dict[task].copy()),movement_dict[task]))

        #     y = KMeans(n_clusters=k, random_state=0).fit(x_original)
        #     marker = y.predict(x_original)
        #     marker_list = [marker_dict[i] for i in marker]
        #     color_list = [color_dict[i] for i in marker]
            
        #     x_original = (x_original-mean_locomotion_state_ )/std_locomotion_state_
        #     x_new = pca.transform(x_original)
        #     all_pca_x = np.vstack((all_pca_x, x_new))

        #     title = fig_name_by_task[task] + ", kmeans clustering, k = " + str(k)
        #     fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"kmeans_"+task+"_"+str(k)+"_no_center.png"
        #     draw_k_means(x_new, focus_explore_diff_dict[task], max_, colormap, marker_list, title, fig_path)

            # with state-time and toy

            # with state-time, toy playtime, and movement

        # print(np.unique(all_marker_list))
        





        # tsne = TSNE(n_components=k, perplexity = 15).fit(all_data)
        for perplexity in [5, 10, 15, 20, 25]:
            x_new = TSNE(n_components=2, perplexity = perplexity).fit_transform(x_state_movement)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+".png"
            title = "All conditions, kmeans clustering, using state and locomotion, T-SNE, k = " + str(k)
            draw_kmeans_for_all(x_new, all_colors_list, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+"_mom.png"
            draw_kmeans_for_all(x_new, all_colors_list, all_marker_mom, ["With caregiver", "Without caregiver"], title, fig_path)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+"_toy.png"
            draw_kmeans_for_all(x_new, all_colors_list, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"],  title, fig_path)

            # state-toy
            x_new = TSNE(n_components=2, perplexity = perplexity).fit_transform(state_toy_normed)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+".png"
            title = "All conditions, kmeans clustering, using state and toy as feature, T-SNE, k = " + str(k)
            draw_kmeans_for_all(x_new, all_colors_list_state_toy, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+"_mom.png"
            draw_kmeans_for_all(x_new, all_colors_list_state_toy, all_marker_mom, ["With caregiver", "Without caregiver"], title, fig_path)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+"_toy.png"
            draw_kmeans_for_all(x_new, all_colors_list_state_toy, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"],  title, fig_path)

            # state-toy-locomotion
            x_new = TSNE(n_components=2, perplexity = perplexity).fit_transform(state_toy_locomotion_normed)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+".png"
            title = "All conditions, kmeans clustering, using state and toy and locomotion as feature, T-SNE, k = " + str(k)
            draw_kmeans_for_all(x_new, all_colors_list_state_toy_locomotion, all_marker_list, list(fig_name_by_task.values()), title, fig_path)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+"_mom.png"
            draw_kmeans_for_all(x_new, all_colors_list_state_toy_locomotion, all_marker_mom, ["With caregiver", "Without caregiver"], title, fig_path)

            fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+str(k)+"_perp_"+str(perplexity)+"_toy.png"
            draw_kmeans_for_all(x_new, all_colors_list_state_toy_locomotion, all_marker_toy, ["Fine-motor toys", "Gross-motor toys"],  title, fig_path)




            
            # for task in tasks:
            #     kmeans = KMeans(n_clusters = k, random_state=0).fit(state_toy_locomotion_normed)
            #     x_original = np.hstack((all_task_freq_dict[task], movement_dict[task]))

            #     colors = kmeans.predict(x_original)
            #     x_new = TSNE(n_components=2, perplexity = perplexity).fit_transform(x_original)

            #     color_list = [color_dict[i] for i in colors]

            #     # x_original = (x_original-mean_)/std_

            #     title = fig_name_by_task[task]


            #     fig_path = "figures/hmm/20210824/n_new_toy_ratio/no_ops_threshold_10/window_size_1.5/5_states/"+"tsne_kmeans_"+task+"_"+str(k)+"_perp_"+str(perplexity)+".png"
            #     draw_tsne(x_new, color_list, title, fig_path)

