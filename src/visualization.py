from itertools import groupby, permutations, combinations
import os
from matplotlib import pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from jupyterthemes import jtplot
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.patches import Rectangle, Patch

np.random.seed(0)

jtplot.style()


def color(color, text):
    s = ' {$\Huge \mathsf{\color{' + str(color) + '}{' + str(text) + '}}$}'
    return s


def save_png(fig, file_path, width, height):
    save_path = file_path
    fig.write_image(str(save_path), width=width, height=height)


def draw_timeline_with_feature(k, df, time_list, shift_time, task,
                               n_toy_switches_list, n_toy_list, n_new_toy_list, fav_toy_ratio_list, toy_iou_list,
                               toy_list,
                               window_size, show=False):
    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']

    df = df.explode('toy')
    df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    data1 = go.Scatter(
        x=df['onset']/60000,
        y=df['toy'],
        mode='markers',
        marker=dict(color='rgba(131, 90, 241, 0)')
    )

    data2 = go.Scatter(
        x=df['offset']/60000,
        y=df['toy'],
        mode='markers',
        marker=dict(color='rgba(131, 90, 241, 0)')
    )

    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'lavender', 'bucket': 'navy'}

    shapes = [dict(
        type='line',
        x0=df['onset'].iloc[i]/60000,
        y0=df['toy'].iloc[i],
        x1=df['offset'].iloc[i]/60000,
        y1=df['toy'].iloc[i],
        line=dict(
            color=colors_dict[df['toy'].iloc[i]],
            width=3
        )
    ) for i in range(len(df['onset']))]

    shapes.append(dict(type="rect",
                       x0=(begin_time+shift_time*60000)/60000,
                       y0=0,
                       x1=time_list[0]/60000,
                       y1=7,
                       line=dict(
                           color="rgba(0, 0, 0, 0)",
                           width=1,
                       ),
                       fillcolor='rgba(255, 0, 0, 0.0)',
                       opacity=.5,
                       ))

    shapes.extend([dict(type="rect",
                        x0=time_list[j-1]/60000,
                        y0=0,
                        x1=time_list[j]/60000,
                        y1=7,
                        line=dict(
                            color="rgba(0, 0, 0, 1)",
                            width=1,
                        ),
                        fillcolor='rgba(255, 0, 0, 0.0)',
                        opacity=.5,
                        ) for j in range(len(time_list)) if j != 0])

    if shift_time == 0:
        title = 'Subject ' + str(k) + ' - ' + task
    else:
        title = 'Subject ' + str(k) + ' - ' + task + \
            ' offset: '+str(shift_time) + ' min'
    layout = go.Layout(
        shapes=shapes,
        title=title
    )

    # Plot the chart
    fig = go.Figure([data1, data2], layout)
    ticktext = [color(v, k) for k, v in colors_dict.items()]
    # print(ticktext)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(tickfont=dict(size=48), gridcolor='black',
                                   tickmode='array', tickvals=list(colors_dict.keys()), ticktext=ticktext),
                      font=dict(size=24),
                      annotations=[
                          dict(
                              x=(time_list[m])/60000-(window_size/4),
                              y=5,
                              text="<br> toy switch rate " +
                              str(round(n_toy_switches_list[m], 2))
                              + "<br> # toy " +
                              str(round(n_toy_list[m], 2))
                              + "<br> new toy rate " +
                              str(round(n_new_toy_list[m], 2))
                              + "<br> dom. toy " +
                              str(round(fav_toy_ratio_list[m], 2))
                              + "<br> toy iou " +
                              str(round(toy_iou_list[m], 2))
                              + "<br> toy_list" +
                              str(toy_list[m]),
                              #   + "<br> new toy list" +
                              #   str(new_toy_list[m])
                              #   + "<br> fav toy list" +
                              #   str(fav_toy_list[m]),
                          ) for m in range(len(time_list))]
                      )
    fig.update_yaxes(categoryorder='category descending')
    fig.update_layout(yaxis_type='category', showlegend=False)
    if show:
        fig.show()
    return fig


def draw_plain_timeline(k, df, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = df['toy'].unique().tolist()
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy'}
    state_color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple', 5: 'chocolate', 6: 'crimson', 7: 'darkolivegreen'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])
   
    
    plt.title('Subject ' + str(k), fontsize = 16)
    plt.xlabel('Minutes', fontsize = 16)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 16)
    plt.grid(False)
    plt.xticks(fontsize = 16)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
     
def draw_plain_timeline_with_feature_discretization(k, df, time_list, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = df['toy'].unique().tolist()
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy'}
    # state_color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple', 5: 'chocolate', 6: 'crimson', 7: 'darkolivegreen'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])


    height = len(toys)
    time_list = np.array(time_list)/60000 - begin_time
    time = time_list[0] - .25
    while time < time_list[-1]:
        ax.add_patch(Rectangle((time, 0), 1.5, height,fill = False, ec = 'b'))
        time += 1.5


    plt.title('Subject ' + str(k), fontsize = 16)
    plt.xlabel('Minutes', fontsize = 16)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 16)
    plt.grid(False)
    plt.xticks(fontsize = 16)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)



def draw_timeline_with_merged_states(k, df, state_list, state_name_list, time_list, state_name, fig_name, show=False):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = df['toy'].unique().tolist()
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy'}
    state_color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple', 5: 'chocolate', 6: 'crimson', 7: 'darkolivegreen'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000 - begin_time, offset_/60000 - begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])
   

    height = len(toys)
    time_list = (np.array(time_list)/60000 - begin_time)
    for idx, pred in enumerate(state_list):
        if idx == 0:
            ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, fc = state_color_dict[pred], fill = True, alpha = 0.1))
        else:
            ax.add_patch(Rectangle((time_list[idx-1], 0), time_list[idx] - time_list[idx-1], height, fc = state_color_dict[pred], fill = True, alpha = 0.1))
            
    # create legend
    # plt.axis('off')
    # [t.set_color(colors_dict[inverse_dict[idx]]) for idx, t in enumerate(ax.xaxis.get_ticklines())]
    # [t.set_color(colors_dict[inverse_dict[idx]]) for idx, t in enumerate(ax.xaxis.get_ticklabels())]

    legend_elements = []
    for state in np.unique(state_list):
        legend_elements.append(Patch(facecolor=state_color_dict[state], edgecolor=state_color_dict[state], label=state_name[state], fill = True, alpha = 0.3))
    
    plt.title('Subject ' + str(k), fontsize = 16)
    plt.xlabel('Minutes', fontsize = 16)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 16)
    plt.grid(False)
    plt.xticks(fontsize = 16)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
        


def draw_distribution(n_features, state_name_dict, feature_vector, pred, task, feature_name, x_ticks_dict, feature_values):
    n_states = len(state_name_dict.keys())
    
    fig, axs = plt.subplots(n_states, n_features, sharex='col', sharey='row',figsize=(14,14))
    state_color_dict = {0:'blue', 1:'green', 2:'purple', 3:'orange', 4:'yellow', 5:'chocolate', 6:'crimson', 7:'darkolivegreen'}
    for f_i in range(len(feature_vector.T)):
        feature = feature_vector.T[f_i]
        all_unique = np.unique(feature)
        # x_labels = [str(int(x_i)) for x_i in all_unique]
        x_labels = x_ticks_dict[f_i]
        x_vals = feature_values[f_i]
        for idx, state in enumerate(list(state_name_dict.keys())):

            final_val = []
            final_height = []
            unique, cnt = np.unique(feature[pred == state], return_counts = True)
            height = cnt/cnt.sum()
            cnt_dict = {k: v for k,v in zip(unique, height)}

            for val in x_vals:
                final_val.append(val)

                if val in cnt_dict.keys():
                    final_height.append(cnt_dict[val])
                else:
                    final_height.append(0)
            axs[idx, f_i].set_ylim(top=1) 
            axs[idx, f_i].bar(final_val, final_height, color = state_color_dict[state])

            axs[idx, f_i].set_xticks(x_vals)
            axs[idx, f_i].set_xticklabels(labels = x_labels, fontsize = 28)

            axs[idx, f_i].set_yticks(np.arange(0,1.1,0.5))
            axs[idx, f_i].set_yticklabels(labels = [str(np.around(y_i,1)) for y_i in np.arange(0,1.1,0.5)],  fontsize = 28)
            axs[idx, 0].tick_params("y", left=True, labelleft=True)
            axs[idx, -1].tick_params("y", right=False, labelright=False)

        axs[-1, f_i].set_xlabel(feature_name[f_i])
    for idx, state in enumerate(list(state_name_dict.keys())): 
        axs[idx, -1].set_ylabel(state_name_dict[state], fontsize = 28)
        axs[idx, -1].yaxis.set_label_position("right")

    plt.suptitle('Emission distribution, task: ' + str(task),  fontsize = 28)
    plt.tight_layout()
    plt.show()

def draw_infant_each_min_matplotlib(focus_cnt, explore_cnt):
    x = np.arange(44)
    tickvals = [x for x in range(41) if x % 2 ==0]
    ticktext = [str(int(x/2)) for x in tickvals]
    x_labels = []
    fig, ax = plt.subplots()#figure(figsize=(6,4))
    explore_plot, = ax.plot(x[:41], explore_cnt[:41], marker = 'v', color = 'green', label = 'Explore states: E-, E+')
    focus_plot, = ax.plot(x[:41], focus_cnt[:41], marker = 'o', color = 'orange', label = 'Focus states: F1, F2, Fset')

    ax.legend(handles=[explore_plot, focus_plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Number of infant')
    ax.set_yticks(np.arange(0, 20, 2))
    ax.set_xticks(np.arange(0, 42, 2))

    ax.set_xticklabels([str(x) for x in np.arange(0,21,1)])
    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]

    plt.show()

def draw_infant_each_min_matplotlib(focus_cnt: list, explore_cnt: list, no_ops_state: list, name):
    x = np.arange(17)
    tickvals = [x for x in range(17) if x % 2 ==0]
    ticktext = [str(int(x/2)) for x in tickvals]
    x_labels = []
    fig, ax = plt.subplots()#figure(figsize=(6,4))
    explore_plot, = ax.plot(x, explore_cnt[:17], marker = 'v', color = 'green', label = 'E, E+')
    focus_plot, = ax.plot(x, focus_cnt[:17], marker = 'o', color = 'orange', label = 'F, F+')
    no_ops_plot, = ax.plot(x, no_ops_state[:17], marker = 'h', color = 'blue', label = 'No_ops')


    ax.legend(handles=[explore_plot, focus_plot, no_ops_plot], bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Number of infant')
    # ax.set_yticks(np.arange(0, 20, 2))
    ax.set_xticks(np.arange(0, 17, 2))

    ax.set_xticklabels([str(x) for x in np.arange(0,len(ticktext),1)])

    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
    ax.set_ylim(top = 35)
    plt.title(name, fontsize = 16)

    plt.show()