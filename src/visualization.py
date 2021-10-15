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
from math import ceil
np.random.seed(0)
import matplotlib.transforms as mtrans
from itertools import chain
jtplot.style()


def color(color, text):
    s = ' {$\Huge \mathsf{\color{' + str(color) + '}{' + str(text) + '}}$}'
    return s

def save_png(fig, file_path, width, height):
    save_path = file_path
    fig.write_image(str(save_path), width=width, height=height)

def draw_comparison_new_merge(k, df_list, y_labels, title):
    plt.style.use('seaborn')

    fig, ax = plt.subplots(ncols= 1, nrows= len(df_list), sharex = True, figsize = (120,15*len(df_list))) 
    
    for idx, df in enumerate(df_list):
        begin_time = df.iloc[0, :]
        begin_time = begin_time.loc['onset']/60000

        df = df.explode('toy')
        df = df.replace({"no_ops": "no_toy"})

        # original_df = original_df.loc[original_df.loc[:, 'toy'] != 'no_ops']
        toys = sorted(df['toy'].unique().tolist(), reverse = True)
        toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
        inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
        # print(toy_dict)
        colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                    'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                    'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy','no_toy': 'slategrey'}
        
        for t in toys:
            onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
            offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
            for onset_, offset_ in zip(onset_list, offset_list):
                ax[idx].plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 15, c = colors_dict[t])
        
        # if 'merge' in df.columns:
        #     roi_onset = df.loc[df.loc[:,'merge'] == 1, 'onset'].to_numpy()/60000-begin_time
        #     roi_offset = df.loc[df.loc[:,'merge'] == 1, 'offset'].to_numpy()/60000-begin_time
        #     time_list = np.sort(np.unique(np.array(roi_onset.tolist() + roi_offset.tolist())))
            # print(time_list)

            # for roi_onset_, roi_offset_ in zip(time_list[:-1], time_list[1:]):
            #     ax[idx].add_patch(Rectangle((roi_onset_, 0), (roi_offset_ - roi_onset_), len(toys), ec = None, fc = 'red', fill = True, alpha = .1))
        ax[idx].grid(b = False, axis = 'y')
        ax[idx].grid(b = True, axis = 'x', which = 'both')
        ax[idx].minorticks_on()
        ax[idx].xaxis.grid(b = True, which = "major", color='black', linestyle='--', linewidth=5, alpha = .5)
        ax[idx].xaxis.grid(b = True, which = "minor", color='grey', linestyle='--', linewidth=5, alpha = .3)
        ax[idx].set_facecolor('white')
        ax[idx].set_yticks(list(toy_dict.values()))
        ax[idx].set_yticklabels(list(toy_dict.keys()), fontsize = 32*len(df_list))

        y_labels = [l for l in ax[idx].yaxis.get_ticklabels()]
        for i in range(len(toy_dict.keys())):
            y_labels[i].set_color(colors_dict[inverse_dict[i]])
        ax[idx].set_ylabel(y_labels[idx])
        
    
    offset_list = df_list[0]['offset'].tolist()
    ax[0].set_title(title, fontdict = {'fontsize' : 36*len(df_list)})
    # ax[0].title.set_titlesize(fontsize = 24*len(df_list))
    # plt.suptitle(title, fontsize = 24*len(df_list))
    ticklist = (np.array(offset_list) - offset_list[0])/60000
    ax[-1].set_xticks(list(range(ceil(max(ticklist)) + 1 )))
    ax[-1].set_xticklabels([str(x) for x in range(int(ceil(max(ticklist))) + 1)], fontsize = 36*len(df_list))
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

    #Get the minimum and maximum extent, get the coordinate half-way between those
    ax = ax.reshape((1,4))
    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(ax.shape).max(axis=1)
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(ax.shape).min(axis=1)
    ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0,1],[y,y], transform=fig.transFigure, color="black")
        fig.add_artist(line)
    plt.xlim(left = 0)
    plt.xlabel('Minutes', fontsize = 36*len(df_list))
    plt.tight_layout()
    plt.savefig('./examples/'+str(k)+'.png')
    plt.close()

def draw_comparison(k, df_list, title, roi_onset, roi_offset):
    plt.style.use('seaborn')

    fig, ax = plt.subplots(ncols= 1, nrows= len(df_list), sharex = True, figsize = (30,15)) 
    plt.suptitle(title, fontsize = 24)
    
    for idx, df in enumerate(df_list):
        begin_time = df.iloc[0, :]
        begin_time = begin_time.loc['onset']/60000

        df = df.explode('toy')
        df = df.replace({"no_ops": "no_toy"})

        # original_df = original_df.loc[original_df.loc[:, 'toy'] != 'no_ops']
        toys = sorted(df['toy'].unique().tolist(), reverse = True)
        toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
        inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
        # print(toy_dict)
        colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                    'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                    'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy','no_toy': 'slategrey'}
        
        for t in toys:
            onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
            offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
            for onset_, offset_ in zip(onset_list, offset_list):
                ax[idx].plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])

        ax[idx].add_patch(Rectangle((roi_onset/60000-begin_time, 0), (roi_offset-roi_onset)/60000, len(toys), ec = 'black', fc = 'red', fill = True, alpha = .1))
        ax[idx].grid(b = False, axis = 'y')
        ax[idx].grid(b = True, axis = 'x', which = 'both')
        ax[idx].minorticks_on()
        ax[idx].xaxis.grid(b = True, which = "major", color='black', linestyle='--', linewidth=1, alpha = .3)
        ax[idx].xaxis.grid(b = True, which = "minor", color='grey', linestyle='--', linewidth=1, alpha = .1)
        ax[idx].set_facecolor('white')
        ax[idx].set_yticks(list(toy_dict.values()))
        ax[idx].set_yticklabels(list(toy_dict.keys()), fontsize = 20)

        y_labels = [l for l in ax[idx].yaxis.get_ticklabels()]
        for i in range(len(toy_dict.keys())):
            y_labels[i].set_color(colors_dict[inverse_dict[i]])

    offset_list = df_list[0]['offset'].tolist()
    ticklist = (np.array(offset_list) - offset_list[0])/60000
    ax[-1].set_xticks(list(range(ceil(max(ticklist)) + 1 )))
    ax[-1].set_xticklabels([str(x) for x in range(int(ceil(max(ticklist))) + 1)], fontsize = 16)
    plt.xlim(left = 0)
    plt.xlabel('Minutes', fontsize = 20)
    plt.tight_layout()
    plt.savefig('./examples/'+str(k)+'.png')
    plt.close()

def draw_plain_timeline(k, df, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.replace({"no_ops":'no_toys'})

    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toys':'slategrey'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])
    
    onset_list = df['onset'].to_numpy()
    ticklist = (np.array(onset_list) - onset_list[0])/60000
    # print(np.ceil(max(ticklist))) 
    plt.title('Subject ' + str(k), fontsize = 20)
    plt.xlabel('Minutes', fontsize = 16)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 16)
    plt.xticks(list(range(ceil(max(ticklist)) + 1)), [str(x) for x in range(int(ceil(max(ticklist))) + 1)], fontsize = 16)

    plt.grid(False)
    plt.minorticks_on()
    # ax.tick_params(axis='x', which='minor', bottom=True)

    ax.xaxis.grid(b = True, which = "major", color='black', linestyle='--', linewidth=1, alpha = .3)
    ax.xaxis.grid(b = True, which = "minor", color='grey', linestyle='--', linewidth=1, alpha = .1)
    plt.xlim(left = 0)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def draw_plain_timeline_with_feature_discretization_compare(k, df_list, time_big_list, features_big_list, gap_size, fav_toy_big_list, fig_name):
    plt.style.use('seaborn')

    fig, ax = plt.subplots(ncols= 1, nrows = len(df_list), sharex = True, figsize = (100,20*len(df_list))) 
    for ax_idx, df in enumerate(df_list):
        time_list, features, fav_toy_list = time_big_list[ax_idx], features_big_list[ax_idx], fav_toy_big_list[ax_idx]
        begin_time = df.iloc[0, :]
        begin_time = begin_time.loc['onset']/60000

        df = df.explode('toy')
        df = df.replace({'no_ops': 'no_toy'})
        # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
        toys = sorted(df['toy'].unique().tolist(), reverse = True)
        toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
        inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
        # print(toy_dict)
        colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                    'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                    'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toy': 'slategrey'}
        # state_color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple', 5: 'chocolate', 6: 'crimson', 7: 'darkolivegreen'}
        
        for t in toys:
            onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
            offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
            for onset_, offset_ in zip(onset_list, offset_list):
                ax[ax_idx].plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])


        height = len(toys)
        time_list = np.array(time_list)/60000 - begin_time
        time = time_list[0] - .25
        # while time < time_list[-1]:
        for idx, _ in enumerate(time_list):
            if fav_toy_list is not None:
                text = "# switches " +\
                        str(features[idx, 0])\
                        + "\n# toys " +\
                        str(features[idx,1])\
                        + "\n# new toys " +\
                        str(features[idx,2])\
                        + "\nfav. toy ratio " +\
                        str(round(features[idx,3], 2))\
                        +"\nfav toys " + fav_toy_list[idx]
                        # +"\n" +', '.join([str(elem) for elem in toy_present_big_list[idx]])
            else:
                text = "# switches " +\
                        str(features[idx, 0])\
                        + "\n# toys " +\
                        str(features[idx,1])\
                        + "\n# new toys " +\
                        str(features[idx,2])\
                        + "\nfav. toy ratio " +\
                        str(round(features[idx,3], 2))\

            if idx == 0:
                ax[ax_idx].add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, ec = 'black', fill = False))
                ax[ax_idx].annotate(text, (time_list[0] - gap_size, height + 1), fontsize = 20*len(df_list), color = 'black',  ha = 'left', va = 'center')

            # elif time_list[idx] - time_list[idx-1] <= gap_size:
            else:
                ax[ax_idx].add_patch(Rectangle((time_list[idx-1], 0), gap_size, height, ec = 'black', fill = False))
                ax[ax_idx].annotate(text, (time_list[idx-1], height + 1), fontsize = 20*len(df_list), color = 'black',  ha = 'left', va = 'center')
        y_labels = [l for l in ax[ax_idx].yaxis.get_ticklabels()]
        for i in range(len(toy_dict.keys())):
            y_labels[i].set_color(colors_dict[inverse_dict[i]])
        ax[ax_idx].set_facecolor('white')
        ax[ax_idx].set_ylim(top = height + 2)  
        ax[ax_idx].set_yticks(list(toy_dict.values()))
        ax[ax_idx].set_yticklabels(list(toy_dict.keys()), fontsize = 24*len(df_list))


    # plt.title('Subject ' + str(k), fontsize = 24*len(df_list))
    ax[0].set_title('Subject ' + str(k), fontdict = {'fontsize' : 24*len(df_list)})
    ax[-1].set_xlabel('Minutes', fontsize = 24*len(df_list))
    plt.grid(False)
    plt.xticks(fontsize = 24*len(df_list))

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    
def draw_plain_timeline_with_feature_discretization(k, df, time_list, features, gap_size, fav_toy_list, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.replace({'no_ops': 'no_toy'})
    # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toy': 'slategrey'}
    
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
    # while time < time_list[-1]:
    for idx, _ in enumerate(time_list):
        if fav_toy_list is not None:
            text = "# switches " +\
                    str(features[idx, 0])\
                    + "\n# toys " +\
                    str(features[idx,1])\
                    + "\n# new toys " +\
                    str(np.round(features[idx,2], 2))\
                    + "\nfav. toy ratio " +\
                    str(round(features[idx,3], 2))\
                    +"\nfav toys " + fav_toy_list[idx]
                    # +"\n" +', '.join([str(elem) for elem in toy_present_big_list[idx]])
        else:
            text = "# switches " +\
                    str(features[idx, 0])\
                    + "\n# toys " +\
                    str(features[idx,1])\
                    + "\n# new toys/# toys " +\
                    str(np.round(features[idx,2], 2))\
                    + "\nfav. toy ratio " +\
                    str(round(features[idx,3], 2))\

        if idx == 0:
            ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, ec = 'black', fill = False))
            ax.annotate(text, (time_list[0] - gap_size, height + 1), fontsize = 16, color = 'black',  ha = 'left', va = 'center')

        elif time_list[idx] - time_list[idx-1] <= gap_size:
            ax.add_patch(Rectangle((time_list[idx-1], 0), gap_size, height, ec = 'black', fill = False))
            ax.annotate(text, (time_list[idx-1], height + 1), fontsize = 16, color = 'black',  ha = 'left', va = 'center')


    plt.title('Subject ' + str(k), fontsize = 24)
    plt.xlabel('Minutes', fontsize = 24)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 24)
    plt.grid(False)
    plt.xticks(fontsize = 24)
    plt.ylim(top = height + 2)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def draw_plain_timeline_with_feature_discretization_to_check(k, df, time_list, features, new_toy_list, gap_size, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.replace({'no_ops': 'no_toy'})
    # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toy': 'slategrey'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])


    height = len(toys)
    _, n_features = features.shape
    time_list = np.array(time_list)/60000 - begin_time
    time = time_list[0] - .25
    # while time < time_list[-1]:
    x_ticks_list = [0]
    for idx, _ in enumerate(time_list):
        text = "# switches " + str(features[idx, 0])\
                + "\n# toys " + str(features[idx,1])\
                + "\n# new toys ratio " + str(round(features[idx,3], 2))\
                + "\nfav toy playtime ratio " + str(round(features[idx,5], 1))\
                # + "\nglobal fav. toy " + str(round(features[idx,6], 2))\
                # +"\n" +'\n'.join([str(elem) for elem in new_toy_list[idx]])
                # + "\n# new toys " + str(features[idx,2])\
                # + "\nnew toy playtime " + str(round(features[idx,4], 2))\


    

        if idx == 0:
            ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, ec = 'black', fill = False))
            ax.annotate(text, (time_list[0] - gap_size, height+.75), fontsize = 16, color = 'black',  ha = 'left', va = 'center')
            x_ticks_list.append(time_list[0]-(time_list[1] - time_list[0]))

        elif time_list[idx] - time_list[idx-1] <= gap_size:
            ax.add_patch(Rectangle((time_list[idx-1], 0), gap_size, height, ec = 'black', fill = False))
            ax.annotate(text, (time_list[idx-1], height+.75), fontsize = 16, color = 'black',  ha = 'left', va = 'center')
            x_ticks_list.append(time_list[idx-1])
    # x_ticks_list.append(time_list[idx])


    # plt.title('Subject ' + str(k), fontsize = 24)
    plt.xlabel('Minutes', fontsize = 24)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 24)
    plt.grid(False)
    plt.xticks(x_ticks_list, fontsize = 24)
    plt.ylim(top = height + 2)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def draw_timeline_with_merged_states(k, df, state_list, time_list, state_name, fig_name, gap_size, state_color_dict):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    df = df.replace({"no_ops":"no_toy"})
    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', "no_toy":'slategrey'}
    # state_color_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple', 5: 'chocolate', 6: 'crimson', 7: 'darkolivegreen'}
    # state_color_dict = {"F++":'blue', "F+":'green', "F":'purple', "E":'orange', "E+":'yellow', 'no_toy':'chocolate', "E++":'crimson', 7:'darkolivegreen'}
    # state_color_dict = {"0":'chocolate',  "1":'green', "2":'purple', "3":'orange', "4":'yellow',  "6":'crimson', "7":'darkolivegreen', "8":'blue'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000 - begin_time, offset_/60000 - begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])
   

    height = len(toys)
    time_list = (np.array(time_list)/60000 - begin_time)
    time_list = np.round(time_list, 2) # numerical error
    if len(state_list) > 1:
        for idx, pred in enumerate(state_list):
            if idx == 0:
                ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, fc = state_color_dict[state_name[pred]], ec = 'black', fill = True, alpha = 0.3))
            elif time_list[idx] - time_list[idx-1] <= gap_size:
                ax.add_patch(Rectangle((time_list[idx-1], 0), gap_size, height, fc = state_color_dict[state_name[pred]], fill = True, ec = 'black', alpha = 0.3))
    else:
        ax.add_patch(Rectangle((time_list[0]-gap_size, 0), gap_size, height, fc = state_color_dict[state_name[state_list[0]]], ec = 'black', fill = True, alpha = 0.3))

    # create legend
    # plt.axis('off')
    # [t.set_color(colors_dict[inverse_dict[idx]]) for idx, t in enumerate(ax.xaxis.get_ticklines())]
    # [t.set_color(colors_dict[inverse_dict[idx]]) for idx, t in enumerate(ax.xaxis.get_ticklabels())]

    legend_elements = []
    for state in np.unique(state_list):
        legend_elements.append(Patch(facecolor=state_color_dict[state_name[state]], edgecolor=state_color_dict[state_name[state]], label=state_name[state], fill = True, alpha = 0.5))
    
    plt.title('Subject ' + str(k), fontsize = 16)
    plt.xlabel('Minutes', fontsize = 16)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 16)
    plt.grid(False)
    plt.xticks(fontsize = 16)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])

    ax.legend(handles=legend_elements, loc='upper right', fontsize = 16)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
        
def draw_distribution(n_features, state_name_dict, feature_vector, pred, task, feature_name, x_ticks_dict, feature_values, state_color_dict, fig_path):
    plt.style.use('seaborn')
    n_states = len(state_name_dict.keys())
    
    fig, axs = plt.subplots(n_states, n_features, sharey='row',figsize=(14,14))
    # state_color_dict = {0:'blue', 1:'green', 2:'purple', 3:'orange', 4:'yellow', 5:'chocolate', 6:'crimson', 7:'darkolivegreen'}
    # state_color_dict = {"F++":'blue', "F+":'green', "F":'purple', "E":'orange', "E+":'yellow', 'no_toy':'chocolate', "E++":'crimson', 7:'darkolivegreen'}

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
            axs[idx, f_i].bar(final_val, final_height, color = state_color_dict[state_name_dict[state]])

            axs[idx, f_i].set_xticks(x_vals)
            axs[idx, f_i].set_xticklabels(labels = x_labels, fontsize = 16)

            axs[idx, f_i].set_yticks(np.arange(0,1.1,0.5))
            axs[idx, f_i].set_yticklabels(labels = [str(np.around(y_i,1)) for y_i in np.arange(0,1.1,0.5)],  fontsize = 28)
            axs[idx, 0].tick_params("y", left=True, labelleft=True)
            axs[idx, -1].tick_params("y", right=False, labelright=False)

            axs[idx, f_i].set_xlabel(feature_name[f_i], fontsize = 26)
    for idx, state in enumerate(list(state_name_dict.keys())): 
        axs[idx, -1].set_ylabel(state_name_dict[state], fontsize = 28)
        axs[idx, -1].yaxis.set_label_position("right")

    plt.suptitle('Emission distribution, ' + str(task),  fontsize = 28)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    
# def draw_infant_each_min_matplotlib(focus_cnt, explore_cnt):
#     x = np.arange(44)
#     tickvals = [x for x in range(41) if x % 2 ==0]
#     ticktext = [str(int(x/2)) for x in tickvals]
#     x_labels = []
#     fig, ax = plt.subplots()#figure(figsize=(6,4))
#     explore_plot, = ax.plot(x[:41], explore_cnt[:41], marker = 'v', color = 'green', label = 'Explore states: E-, E+')
#     focus_plot, = ax.plot(x[:41], focus_cnt[:41], marker = 'o', color = 'orange', label = 'Focus states: F1, F2, Fset')

#     ax.legend(handles=[explore_plot, focus_plot], bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.set_xlabel('Minutes')
#     ax.set_ylabel('Number of infant')
#     ax.set_yticks(np.arange(0, 20, 2))
#     ax.set_xticks(np.arange(0, 42, 2))

#     ax.set_xticklabels([str(x) for x in np.arange(0,21,1)])
#     [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]

#     plt.show()

def draw_infant_each_min_matplotlib(focus_cnt: list, explore_cnt: list, no_ops_state: list, name, file_name):
    x = np.arange(17)
    tickvals = [x for x in range(17) if x % 2 ==0]
    ticktext = [str(int(x/2)) for x in tickvals]
    x_labels = []
    fig, ax = plt.subplots(figsize=(10,6))
    focus_plot, = ax.plot(x, focus_cnt[:17], marker = 'o', color = 'orange', label = 'F states (1, 2)')
    explore_plot, = ax.plot(x, explore_cnt[:17], marker = 'v', color = 'green', label = 'E states (3, 4)')
    no_ops_plot, = ax.plot(x, no_ops_state[:17], marker = 'h', color = 'blue', label = 'No_ops')


    ax.legend(handles=[focus_plot, explore_plot, no_ops_plot], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 20)
    ax.set_xlabel('Minutes', fontsize = 20)
    ax.set_ylabel('Number of infant', fontsize = 20)
    # ax.set_yticks(np.arange(0, 20, 2))
    ax.set_xticks(np.arange(0, 17, 2))

    ax.set_xticklabels([str(x) for x in np.arange(0,len(ticktext),1)], fontsize = 20)
    # ax.set_yticklabels([str(x) for x in np.arange(0,len(ticktext),1)], fontsize = 20)
    plt.yticks(fontsize=20)

    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
    ax.set_ylim(top = 35)
    plt.title(name, fontsize = 22)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def draw_toy_state(state_name_dict, toy_to_pred_dict, toy_list, toy_colors_dict, name, fig_path, indv = None):
    if indv:
        fig = plt.figure(figsize= (12,15))
    else:
        fig = plt.figure(figsize= (15,8))
    for x_loc, state in enumerate(state_name_dict.values()):
        if x_loc > 0:
            current_state_dict_stationary_toy = {k: toy_to_pred_dict[k] for k in toy_to_pred_dict.keys() if state in k}
            for idx, toy in enumerate(toy_list):
                key = (state, toy)
                if key not in toy_to_pred_dict.keys():
                    val = 0
                else:
                    val = current_state_dict_stationary_toy[key]
                if x_loc == 1:
                    plt.bar(x_loc*10 + idx, val, label = toy, color = toy_colors_dict[toy])
                else:
                    plt.bar(x_loc*10 + idx, val, color = toy_colors_dict[toy])
    plt.xticks(np.arange(12.5, 10*len(list(state_name_dict.values())), 10), list(state_name_dict.values())[1:], fontsize = 28)
    plt.ylabel("% time in each state playing with toys", fontsize = 28)

    plt.xlabel('States', fontsize = 28)
    # plt.xtick(range(len(state_name_dict)-1), list(state_name_dict.values()))
    plt.ylim(top = 0.6)
    plt.yticks(fontsize = 28)
    plt.title(name, fontsize = 28)

    plt.legend(fontsize = 28)
    plt.savefig(fig_path)
    plt.close()

def draw_toy_state_with_std(state_name_dict, toy_to_pred_dict, toy_list, std_dict, toy_colors_dict, name, fig_path, indv = None):
    plt.style.use('seaborn')
    if indv:
        fig = plt.figure(figsize= (15,15), facecolor='white')
    else:
        fig = plt.figure(figsize= (15,8), facecolor='white')
    for x_loc, state in enumerate(state_name_dict.values()):
        if x_loc > 0:
            current_state_dict_stationary_toy = {k: toy_to_pred_dict[k] for k in toy_to_pred_dict.keys() if state in k}
            for idx, toy in enumerate(toy_list):
                key = (state, toy)
                if key not in toy_to_pred_dict.keys():
                    val = 0
                else:
                    val = current_state_dict_stationary_toy[key]
                if x_loc == 1:
                    plt.bar(x_loc*10 + idx, val, label = toy, color = toy_colors_dict[toy])
                else:
                    plt.bar(x_loc*10 + idx, val, color = toy_colors_dict[toy])
                if val != 0:
                    plt.errorbar(x_loc*10+ idx, val, [[0],[std_dict[state][toy]]], barsabove = True, color = 'dimgray')

    plt.xticks(np.arange(12.5, 10*len(list(state_name_dict.values())), 10), list(state_name_dict.values())[1:], fontsize = 28)
    plt.ylabel("% time in each state playing with toys", fontsize = 28)

    plt.xlabel('States', fontsize = 28)
    # plt.xtick(range(len(state_name_dict)-1), list(state_name_dict.values()))
    plt.ylim(top = 1)
    plt.yticks(np.arange(0, 1.1,.2), [str(i) for i in np.arange(0, 110,20)], fontsize = 28)
    plt.title(name, fontsize = 28)
    plt.grid(False)
    if not indv:
        plt.legend(fontsize = 20, loc = "upper right")
    # plt.tight_layout()
    plt.savefig(fig_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

def draw_state_distribution(flatten_pred_dict, n_states, state_name_dict, title, state_color_dict, file_path):
    val, cnt = np.unique(np.array(flatten_pred_dict).astype(int), return_counts = True)
    pct = cnt/cnt.sum()
    fig = plt.figure(figsize = (12,10))
    plt.style.use('seaborn')
    # state_color_dict = {"F++":'blue', "F+":'green', "F":'purple', "E":'orange', "E+":'yellow', 'no_toy':'chocolate', "E++":'crimson', 7:'darkolivegreen'}
    # state_color_dict = {"0":'chocolate',  "1":'green', "2":'purple', "3":'orange', "4":'yellow',  "6":'crimson', "7":'darkolivegreen', "8":'blue'}

    task_state_pct = {v: pct[idx] for idx, v in enumerate(val)}
    for i in range(n_states):
        if i not in task_state_pct.keys():
            task_state_pct[i] = 0
    for idx, state in enumerate(list(state_name_dict.keys())):
        plt.bar(idx, task_state_pct[state], color = state_color_dict[state_name_dict[state]])

    plt.xticks(range(n_states), list(state_name_dict.values()), fontsize = 26)
    plt.ylabel("Pct. time spent in each state, all subjects", fontsize = 24)
    locs, labels = plt.yticks()
    # labels = [str(100*float(x.get_text())) for x in labels]
    plt.yticks(np.arange(0, 0.9, 0.1), [str(int(x*100)) for x in np.arange(0, 0.9, 0.1)], fontsize = 26)
    plt.xlabel("States", fontsize = 26)

    plt.ylim(top = .8)
    plt.title(title, fontsize = 32)
    plt.savefig(file_path)
    plt.close()

def draw_mean_state_locotion_across_conditions(data_dict, task_list, condition_name, n_states,ylabel, title, figname):
    """
    data_dict: dict()
    Please make sure the states are ordered
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize = (12,14))
    task_edge_color = {"MPS": 'r','MPM': 'b', "NMS": 'r', 'NMM': 'b'}
    task_face_color = {"MPS": 'r','MPM': 'b', "NMS": 'none', 'NMM': 'none'}
    task_linestyle = {"MPS": '-','MPM': '-', "NMS": "--", 'NMM': "--"}
    # task_fill ={"MPS": 'full','MPM': 'full', "NMS": 'none', 'NMM': 'none'}
    
    for state_position in range(n_states):
        if state_position == 1:
            for task_idx, task in enumerate(task_list):
                if len(data_dict[task][state_position]) == 0:
                    ax.scatter([state_position*6 + task_idx], 0, edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                else:
                    ax.scatter([state_position*6 + task_idx], np.mean(data_dict[task][state_position]), label = condition_name[task],edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                    ax.errorbar(state_position*6 + task_idx, np.mean(data_dict[task][state_position]), yerr = np.std(data_dict[task][state_position]), ecolor = task_edge_color[task], linestyle = task_linestyle[task], elinewidth = 1)
        else:
            for task_idx, task in enumerate(task_list):
                if len(data_dict[task][state_position]) == 0:
                    ax.scatter([state_position*6 + task_idx], 0, edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                else:
                    ax.scatter([state_position*6 + task_idx], np.mean(data_dict[task][state_position]), edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                    ax.errorbar(state_position*6 + task_idx, np.mean(data_dict[task][state_position]), yerr = np.std(data_dict[task][state_position]), ecolor = task_edge_color[task], linestyle = task_linestyle[task], elinewidth = 1)
    
    ax.legend(loc = 2, fontsize = 20)
    ax.set_xticks(np.arange(1.5, 1.5+6*n_states, 6))
    ax.set_xticklabels([str(i) for i in range(n_states)],fontsize = 20)
    ax.set_xlabel("States", fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    plt.yticks(fontsize = 20)

    ax.set_facecolor('white')
    # ax.set_ylim(bottom =0)
    # plt.tight_layout()
    plt.title(title, fontsize = 20)
    plt.savefig(figname)
    plt.close()

def draw_mean_state_locotion_across_conditions_separate_mean_std(mean_dict, std_dict, task_list, condition_name, n_states,ylabel, title, figname):
    """
    data_dict: dict()
    Please make sure the states are ordered
    """
    plt.style.use('seaborn')
    offset = 10
    fig, ax = plt.subplots(figsize = (30,16))
    task_edge_color = {"MPS": 'r','MPM': 'b', "NMS": 'r', 'NMM': 'b'}
    task_face_color = {"MPS": 'r','MPM': 'b', "NMS": 'none', 'NMM': 'none'}
    task_linestyle = {"MPS": '-','MPM': '-', "NMS": "--", 'NMM': "--"}
    
    for state_position in range(n_states):
        if state_position == 1:
            for task_idx, task in enumerate(task_list):
                if len(mean_dict[task][state_position]) == 0:
                    ax.scatter([state_position*offset + task_idx], 0, edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                else:
                    ax.scatter([state_position*offset + task_idx], np.mean(mean_dict[task][state_position]), label = condition_name[task],edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                    ax.errorbar(state_position*offset + task_idx, np.mean(mean_dict[task][state_position]), yerr = [[0],[std_dict[task][state_position]]], ecolor = task_edge_color[task], linestyle = task_linestyle[task], elinewidth = 1)
        else:
            for task_idx, task in enumerate(task_list):
                if len(mean_dict[task][state_position]) == 0:
                    ax.scatter([state_position*offset + task_idx], 0, edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                else:
                    ax.scatter([state_position*offset + task_idx], np.mean(mean_dict[task][state_position]), edgecolors=task_edge_color[task], facecolors = task_face_color[task], linewidths = 2) 
                    ax.errorbar(state_position*offset + task_idx, np.mean(mean_dict[task][state_position]), yerr = [[0],[std_dict[task][state_position]]], ecolor = task_edge_color[task], linestyle = task_linestyle[task], elinewidth = 1)
    
        if state_position != n_states - 1:
            ax.axvline(state_position*offset + offset//3 + 4, color = 'grey', alpha = 0.1)

    plt.grid(False)
    # ax.xaxis.grid(b = True, which = "minor", color='grey', linestyle='--', linewidth=1, alpha = .5)
    ax.legend(loc = 2, fontsize = 24)
    ax.set_xticks(np.arange(1.5, 1.5+offset*n_states, offset))
    ax.set_xticklabels([str(i) for i in range(n_states)],fontsize = 28)
    # ax.set_yticklabels()
    ax.set_xlabel("States", fontsize = 28)
    ax.set_ylabel(ylabel, fontsize = 28)
    plt.yticks(fontsize = 28)

    ax.set_facecolor('white')
    # ax.set_ylim(bottom =0)
    # plt.tight_layout()
    plt.title(title, fontsize = 28)
    plt.savefig(figname)
    plt.close()

def draw_timeline_with_prob_to_check(k, df, state_list, time_list, state_name, gap_size, state_color_dict, prob_list, shift, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000

    df = df.explode('toy')
    df = df.replace({'no_ops': 'no_toy'})
    # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toy': 'slategrey'}
    
    fig, ax = plt.subplots(figsize = (20,8))
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])

    if shift == 0.5:
        font_size = 16
    else:
        font_size = 12

    height = len(toys)
    time_list = np.array(time_list)/60000 - begin_time
    time = time_list[0] - .25
    # while time < time_list[-1]:
    if len(time_list) > 1:
        for idx, _ in enumerate(time_list):
            highest_states = prob_list[idx].argsort()[-2:][::-1]
            text=str(state_name[highest_states[0]]) +' '+ str(np.round(prob_list[idx][highest_states[0]], 2)) +\
                "\n"+ str(state_name[highest_states[1]]) +' '+ str(np.round(prob_list[idx][highest_states[1]], 2)) 

            if idx == 0:
                ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height, fc = state_color_dict[state_list[idx]], ec = 'black', fill = True, alpha = .3))
                ax.annotate(text, (time_list[0] - gap_size, height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')

            elif time_list[idx] - time_list[idx-1] <= gap_size:
                ax.add_patch(Rectangle((time_list[idx-1], 0), gap_size, height, fc = state_color_dict[state_list[idx]], ec = 'black', fill = True, alpha = .3))
                ax.annotate(text, (time_list[idx-1], height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')
    else:
        highest_states = prob_list[0].argsort()[-2:][::-1]
        text=str(highest_states[0]) +' '+ str(np.round(prob_list[0][highest_states[0]], 2)) +\
            "\n"+ str(highest_states[1]) +' '+ str(np.round(prob_list[0][highest_states[1]], 2)) 
        ax.add_patch(Rectangle((time_list[0]-gap_size, 0), gap_size, height, fc = state_color_dict[state_list[0]], ec = 'black', fill = True, alpha = 0.3))
        ax.annotate(text, (time_list[0] - gap_size, height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')

    plt.title('Subject ' + str(k), fontsize = 24)
    plt.xlabel('Minutes', fontsize = 24)
    plt.yticks(list(toy_dict.values()), list(toy_dict.keys()), fontsize = 24)
    plt.grid(False)
    plt.xticks(fontsize = 24)
    plt.ylim(top = height + 2)
    y_labels = [l for l in ax.yaxis.get_ticklabels()]

    for i in range(len(toy_dict.keys())):
        y_labels[i].set_color(colors_dict[inverse_dict[i]])
    legend_elements = []
    for state in np.unique(state_list):
        legend_elements.append(Patch(facecolor=state_color_dict[state], edgecolor=state_color_dict[state], label=str(state), fill = True, alpha = 0.5))
    ax.legend(handles=legend_elements, loc='upper right', fontsize = 16)
    
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def draw_timeline_with_prob_to_compare(title, df, list_of_state_list, list_of_time_list, state_name, gap_size, state_color_dict, list_of_prob_list, shift, fig_name):
    plt.style.use('seaborn')

    begin_time = df.iloc[0, :]
    begin_time = begin_time.loc['onset']/60000
    plt.suptitle(title, fontsize = 24)

    df = df.explode('toy')
    df = df.replace({'no_ops': 'no_toy'})
    # df = df.loc[df.loc[:, 'toy'] != 'no_ops']
    toys = sorted(df['toy'].unique().tolist(), reverse = True)
    toy_dict = {t: y_coor for y_coor, t in enumerate(toys)}
    inverse_dict = {y_coor: t for t, y_coor in toy_dict.items()}
    # print(toy_dict)
    colors_dict = {'bricks': 'blue', 'pig': 'orange', 'popuppals': 'green', 'xylophone': 'red', 'shape_sorter': 'skyblue',
                   'shape_sorter_blocks': 'salmon', 'broom': 'purple', 'clear_ball': 'teal', 'balls': 'cadetblue',
                   'food': 'chocolate', 'grocery_cart': 'dodgerblue', 'stroller': 'darkblue', 'bucket': 'navy', 'no_toy': 'slategrey'}
    
    fig, axs = plt.subplots(nrows = len(list_of_state_list), ncols = 1, figsize = (5*len(list_of_state_list),8*len(list_of_state_list)//2), sharex = True)
    # plt.suptitle(title)
    for t in toys:
        onset_list = df.loc[df.loc[:, 'toy'] == t, 'onset'].tolist()
        offset_list = df.loc[df.loc[:, 'toy'] == t, 'offset'].tolist()
        data = []
        for onset_, offset_ in zip(onset_list, offset_list):
            for ax in axs:
                ax.plot((onset_/60000- begin_time, offset_/60000- begin_time), (toy_dict[t], toy_dict[t]),  linewidth = 5, c = colors_dict[t])

    if shift == 0.5:
        font_size = 16
    else:
        font_size = 12

    height = len(toys)
    # while time < time_list[-1]:
    for shift_idx, ax in enumerate(axs):
        # time = time_list[0] - .25
        state_list = list_of_state_list[shift_idx]
        time_list = np.array(list_of_time_list[shift_idx])/60000 - begin_time
        prob_list = list_of_prob_list[shift_idx]
        if len(time_list) > 1:
            for idx, _ in enumerate(time_list):
                highest_states = prob_list[idx].argsort()[-2:][::-1]
                # text=str(state_name[highest_states[0]]) +' '+ str(np.round(prob_list[idx][highest_states[0]], 2)) +\
                #     "\n"+ str(state_name[highest_states[1]]) +' '+ str(np.round(prob_list[idx][highest_states[1]], 2)) 

                if idx == 0:
                    ax.add_patch(Rectangle((time_list[0]-(time_list[1] - time_list[0]), 0), time_list[1] - time_list[0], height,  fc = state_color_dict[state_list[idx]], ec = 'black', fill = True, alpha = .3))
                    # ax.annotate(text, (time_list[0] - gap_size, height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')

                elif time_list[idx] - time_list[idx-1] <= gap_size:
                    ax.add_patch(Rectangle((time_list[idx-1], 0), gap_size, height,  fc = state_color_dict[state_list[idx]], ec = 'black', fill = True, alpha = .3))
                    # ax.annotate(text, (time_list[idx-1], height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')
        else:
            highest_states = prob_list[0].argsort()[-2:][::-1]
            text=str(state_name[highest_states[0]]) +' '+ str(np.round(prob_list[0][highest_states[0]], 2)) +\
                "\n"+ str(state_name[highest_states[1]]) +' '+ str(np.round(prob_list[0][highest_states[1]], 2)) 
            ax.add_patch(Rectangle((time_list[0]-gap_size, 0), gap_size, height, fc = state_color_dict[state_list[0]], ec = 'black', fill = True, alpha = .3))
            # ax.annotate(text, (time_list[0] - gap_size, height + .5), fontsize = font_size, color = 'black',  ha = 'left', va = 'center')
    

        
        ax.set_yticks(list(toy_dict.values()))
        ax.set_yticklabels(list(toy_dict.keys()), fontsize = 20)
        y_labels = [l for l in ax.yaxis.get_ticklabels()]
        # print(y_labels)
        # print(len(toy_dict.keys()))
        for i in range(len(y_labels)):
            y_labels[i].set_color(colors_dict[inverse_dict[i]])
        ax.set_ylim(top = height + 2)
        ax.set_facecolor('white')
        
        legend_elements = []
        # all_states = list(chain.from_iterable(list_of_state_list))
        for state in np.unique(state_list):
            legend_elements.append(Patch(facecolor=state_color_dict[state], edgecolor=state_color_dict[state], label=state, fill = True, alpha = 0.5))
        ax.legend(handles=legend_elements, loc='upper right', fontsize = 16)
    
    
    axs[-1].set_xlabel('Minutes', fontsize = 16)
    plt.grid(False)
    plt.xticks(fontsize = 18)
   
    
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()