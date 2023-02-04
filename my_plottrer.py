import pandas as pd
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = './log'


COLORS = [ 'red','salmon', 'brown','green', 'salmon','cyan', 'magenta','darkred',  'yellow', 'black', 'purple', 'pink',
          'teal',  'lightblue', 'orange', 'lavender', 'turquoise','lime',
        'darkgreen', 'tan',  'gold']

# COLORS_map = {'PPO':'steelblue', 'LPPO':'forestgreen', 'LSAC':'brown', 'SAC':'red','SSAC':'olivedrab','SPPO':'gold',
#           'LAC':'red',
#           }
COLORS_map = {
    #fig1
    'Deep Koopman':'deepskyblue',
    'Stochastic Deep Koopman':'red',
    'DKO': 'deepskyblue',
    'DSKO': 'red',
    'MLP': 'purple',

    'DeSKO': 'red',
    'reference':'salmon',
    #fig2
    'RLAC':'red',
    'RARL':'green',
    'SAC':'limegreen',
    'LAC':'red',
    'SPPO':'blue',
    'RLAC-finite_horizon':'brown',
    'MPC':'salmon',
    'SSAC':'magenta',
    'LQR':'brown',
    #fig3
    'CPO-0':'magenta',
    'CPO-1':'green',
    'CPO-10':'deepskyblue',
    'LCPO':'red',
    #fig4

    'N=5':'brown',
    'N=10':'red',
    'N=15':'darkorange',
    'N=20':'gold',
    'infinite horizon':'yellow',
    }

label_fontsize = 10
tick_fontsize = 14
linewidth = 3
markersize = 10





def read_csv(fname):
    return pd.read_csv(fname, index_col=None, comment='#')

def load_results(args, alg_list, contents, env,rootdir=ROOT_DIR):
    # if isinstance(rootdir, str):
    #     rootdirs = [osp.expanduser(rootdir)]
    # else:
    #     dirs = [osp.expanduser(d) for d in rootdir]
    results = {}
    for name in env:
        results[name] = {}
    exp_dirs = os.listdir(rootdir)

    for exp_dir in exp_dirs:

        if exp_dir in env:

            exp_path = os.path.join(rootdir, exp_dir)
            alg_dirs = os.listdir(exp_path)
            for alg_dir in alg_dirs:

                if alg_dir in alg_list:
                    alg_path = os.path.join(exp_path, alg_dir)
                    if args['data'] == 'training':
                        result = read_training_data(alg_path, content)
                    else:
                        result = read_eval_data(env,alg_path, args['eval_content'], content,alg_dir)

                    results[exp_dir][alg_dir] = result

                else:
                    continue


    return results

def read_training_data(alg_path, contents):
    trial_dirs = os.listdir(alg_path)
    trials = []
    result = {}
    min_length = 1e10
    for trial_dir in trial_dirs:
        if trial_dir not in args['plot_list']:
            continue
        full_path = os.path.join(alg_path, trial_dir)
        try:
            trials.append(read_csv(full_path + '/progress.csv'))
        except pd.errors.EmptyDataError:
            continue
        serial_length = len(trials[-1][contents[0]])
        if serial_length < min_length:
            min_length = serial_length
    for key in contents:
        try:
            summary = [trial[key][:min_length] for trial in trials]
        except KeyError:
            continue

        result[key] = np.mean(summary, axis=0)
        std = np.std(summary, axis=0)
        result[key + 'max'] = result[key] + std
        result[key + 'min'] = result[key] - std
    try:
        result['epoch'] = trials[0]['epoch'][:min_length]
    except IndexError:
        print('index error')
    return result


def read_eval_data(env,alg_path, eval_list, content,alg_dir):

    # under the 'eval' directory

    path = alg_path+ '/eval'
    evals = os.listdir(path)
    result = {}
    for eval_content in eval_list:

        if eval_content not in evals:
            print(' '.join([alg_path, 'didn\'t do the experiment on', eval_content]))
            continue

        full_path = os.path.join(path, eval_content)
        try:
            data = read_csv(full_path + '/progress.csv')
        except pd.errors.EmptyDataError:
            print(alg_path + 'reading csv failed')
            continue
        if 'impulse' in eval_content or 'disturbance' in eval_content or 'dynamic' in eval_content:
            summary = collect_line(env,eval_content, data, content,alg_dir)
        else:
            summary = collect_grid(eval_content, data, content)

        result[eval_content] = summary
    return result


def collect_grid(eval_content, data, content_list):

    # collect the grid eval data
    x_name, y_name = eval_content.split('-')
    x_values = data[x_name].values
    x_length = len(np.unique(x_values))
    X = np.reshape(x_values, [x_length, int(len(x_values)/x_length)])

    y_values = data[y_name].values
    y_length = len(np.unique(y_values))
    Y = np.reshape(y_values, [int(len(y_values) / y_length), y_length])

    measure={}
    upper_bound = {}
    lower_bound = {}
    for content in content_list:
        measure[content] = np.reshape(data[content].values, [Y.shape[0], Y.shape[1]])
        if content == 'return':
            upper_bound[content] = 150
            lower_bound[content] = 30
        elif content == 'death_rate':
            upper_bound[content] = 100
            lower_bound[content] = 0
        elif content == 'average_length':
            upper_bound[content] = 250
            lower_bound[content] = 200
        elif content == 'return_std':
            upper_bound[content] = 30
            lower_bound[content] = 10

    summary = {'x': X, 'y': Y, 'measure': measure, 'x_name': x_name, 'y_name': y_name,
               'upper_bound':upper_bound, 'lower_bound':lower_bound }

    return summary


def collect_line(env, eval_content, data, content_list,alg_dir):
    measure = {}
    if eval_content == 'impulse'or eval_content == 'constant_impulse':
        x_name = 'magnitude'
    elif 'disturbance' in eval_content:
        x_name = 'frequency'
    elif 'dynamic' in eval_content:
        x_name = 't'
        measure['reference'] = data['reference'].values
    else:
        x_name = eval_content
    if (alg_dir=='SAC_cost-jacob') & (eval_content== 'dynamic'):

        X= np.arange(1,measure['reference'].shape[0]+1,1,int)
    else:
        X = data[x_name].values
    stds = {}
    for content in content_list:
        if alg_dir=='SAC_cost-jacob':
            try:
                measure[content] = data[content].values
                stds[content] = data[content+'_std'].values
            except KeyError:
                measure[content] = data.values[:,0]

                stds[content] = data.values[:,2]
                # stds[content] = data['reference'].values
                continue
        elif ('trunk_arm_sim' in env) & (alg_dir=='Koopman_v10-jacob'):
            try:
                measure[content] = data.values[:, 6]
                stds[content] = data[content + '_std'].values
            except KeyError:
                measure[content] = data.values[:, 6]

                stds[content] = data.values[:, 1]
                # stds[content] = data['reference'].values
                continue
        elif (('oscillator' in env) or ('oscillator_observation_noise' in env) or ('oscillator_process_noise' in env)) & ( (alg_dir == 'Koopman_v10-jacob') or (alg_dir == 'Koopman_v2-jacob')):
            try:
                measure[content] = data[content].values
                stds[content] = data[content + '_std'].values
            except KeyError:
                measure[content] = data.values[:, 1]

                stds[content] = data.values[:, 5]
                # stds[content] = data['reference'].values
                continue
        else:
            try:
                measure[content] = data[content].values
                stds[content] = data[content + '_std'].values
            except KeyError:
                measure[content] = data.values[:, 0]

                stds[content] = data.values[:, 1]
                # stds[content] = data['reference'].values
                continue

    return {'x': X, 'measure': measure, 'std': stds, 'x_name': x_name, }

def convert(result):

    list = []
    # for i in range(result.size):
    #     no_bracket = result[i].strip("[]")
    #     string = no_bracket.split()
    #
    #     for item in string:
    #         list.append(float(item))
    # list = np.array(list)

    no_bracket = result.strip("[]")
    string = no_bracket.split()

    for item in string:
        list.append(float(item))
    list = np.mat(list)
    return list

def lambdaconvert(result):

    list = []
    # for i in range(result.size):
    #     no_bracket = result[i].strip("[]")
    #     string = no_bracket.split()
    #
    #     for item in string:
    #         list.append(float(item))
    # list = np.array(list)

    # no_bracket = result.strip("[]")
    string = result.split()

    for item in string:
        list.append(float(item))
    list = np.mat(list)
    return list

def floatmatrix(result):

    matrix = np.empty([1,convert(result[0]).shape[1]])

    for i in range(result.size):
        if i == 0:
            matrix = convert(result[i])


        else:
            matrix = np.concatenate((matrix, convert(result[i])))
    matrix = np.array(matrix)
    return matrix

def lambdamatrix(result):

    matrix = np.empty([1,1])

    for i in range(result.size):
        if i == 0:
            matrix = lambdaconvert(result[i])
        else:
            matrix = np.concatenate((matrix, lambdaconvert(result[i])))
    matrix = np.array(matrix)
    return matrix






def plot_training_results(results, alg_list, contents, figsize = None):
    if not args['formal_plot']:
        nrows = len(contents)
        ncols = len(results)
        figsize = figsize or (6, 6)
        f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)
    if args['formal_plot']:
        width = linewidth
        font = 20
    else:
        width = 1
        font = 14
    # plot ep rewards
    content_index = 0
    for content in contents:
        exp_index = 0
        for exp in results.keys():
            min_length = 1e10
            if not args['formal_plot']:
                ax = axarr[content_index][exp_index]
            else:
                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111)
            algs = list(results[exp].keys())
            algs.sort(reverse=False)
            for alg in algs:
                try:
                    result = results[exp][alg]
                except KeyError:
                    continue
                color_index = list(results[exp].keys()).index(alg)
                length = len(result['epoch'])
                sorted_alg_name = sort_alg_name(alg,args['formal_plot'])
                if args['formal_plot'] and sorted_alg_name  in COLORS_map.keys():
                    color = COLORS_map[sorted_alg_name ]
                else:
                    color = COLORS[color_index]
                ax.plot(result['epoch'], result[content], color=color, label=sorted_alg_name, linewidth=width)

                ax.fill_between(result['epoch'], result[content + 'min'], result[content + 'max'],
                                color=color, alpha=.1)
                if length < min_length:
                    min_length = length
            # if exp_index==0:
            # plt.ylabel(CONTENT_YLABEL[content_index],fontsize=label_fontsize)
            if 'lambda' not in content:
                ax.legend(fontsize=font, loc=2, fancybox=False, shadow=False)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            ax.grid(True)
            ax.set_xlabel('epoch')
            ax.set_ylabel(content)
            if 'ylim' in args.keys():
                plt.ylim(args['ylim'][0], args['ylim'][-1])

            # fig = plt.gcf()
            # fig.set_size_inches(9, 6)
            if args['formal_plot']:
                plt.xlim(0, result['epoch'].values[min_length-1])
                plt.savefig('-'.join([exp, 'training', content]) + '.pdf')
                plt.show()
            exp_index += 1
        content_index += 1
    if not args['formal_plot']:
        plt.show()
    return

def plot_mesh(results, alg_list, eval_contents, measure_list, figsize = None):
    if len(eval_contents) == 0:
        return
    # plot ep rewards
    for eval_name in eval_contents:
        if not args['formal_plot']:
            nrows = len(measure_list)
            ncols = len(results)
            figsize = figsize or (3*ncols, 3*nrows)
            fig, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)
        for content_index, content in enumerate(measure_list):



            algs = list(results.keys())
            algs.sort(reverse=True)
            for alg_index, alg in enumerate(algs):
                if not args['formal_plot']:
                    ax = axarr[content_index][alg_index]
                else:
                    fig = plt.figure(figsize=(9, 6))
                    ax = fig.add_subplot(111)
                try:
                    result = results[alg][eval_name]
                except KeyError:
                    continue
                # cm = ['RdBu_r', 'viridis']
                cm = plt.cm.get_cmap('RdBu_r')
                try:
                    img = ax.pcolormesh(result['x'], result['y'], result['measure'][content], cmap=cm,
                                        vmin=result['lower_bound'][content],vmax=result['upper_bound'][content], edgecolors='face')
                except TypeError:
                    print(' '.join([alg,'x and y not compatible']))
                    continue
                cbar = fig.colorbar(img, ax=ax)
                # cbar.set_label('$T_B(K)$', fontdict=14)
                cbar.set_ticks(np.linspace(result['lower_bound'][content],result['upper_bound'][content],5))
                ax.set_xlabel(result['x_name'])
                ax.set_ylabel(result['y_name'])
                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)

                if args['formal_plot']:
                    plt.savefig('-'.join([eval_name, content, alg]) + '.pdf')
                    plt.show()

                else:
                    ax.set_title(alg, fontsize=label_fontsize)

        if not args['formal_plot']:
            plt.show()

    return


def plot_line(env,results, alg_list, eval_contents, measure_list, exp, figsize = None):
    if len(eval_contents) == 0:
        return
    if args['formal_plot']:
        width = linewidth
        font = 20
    else:
        width = 1
        font = 14
    if not args['formal_plot']:
        nrows = len(measure_list)
        ncols = len(eval_contents)
        figsize = figsize or (18, 6)
        fig, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)
    # plot ep rewards
    for eval_index, eval_name in enumerate(eval_contents):

        for content_index, content in enumerate(measure_list):

            algs = list(results.keys())
            #algs.sort()
            algs = sorting_alg(algs)
            if not args['formal_plot']:
                ax = axarr[content_index][eval_index]
            else:
                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111)
            for alg in algs:

                try:
                    result = results[alg][eval_name]
                except KeyError:

                    continue

                color_index = list(results.keys()).index(alg)
                sorted_alg_name = sort_alg_name(alg, args['formal_plot'])
                if args['formal_plot'] and sorted_alg_name in COLORS_map.keys():
                    color = COLORS_map[sorted_alg_name]
                else:
                    color = COLORS[color_index]
                if 'trunk_arm_sim' not in env:
                    try:
                        ax.plot(result['x'], result['measure'][content], color=color, label=sorted_alg_name, linewidth=width)
                        ax.fill_between(result['x'], result['measure'][content] - result['std'][content],
                                        result['measure'][content] + result['std'][content],
                                        color=color, alpha=.1)
                    except KeyError:
                        print(' '.join(['failed in drawing the', content,'of', alg]))
                        continue
                else:
                    if alg == 'SAC_cost-jacob':
                        try:
                            ax.plot(result['x'], result['measure'][content], color=color, label=sorted_alg_name, linewidth=width)
                            ax.fill_between(result['x'], result['measure'][content] - result['std'][content],
                                            result['measure'][content] + result['std'][content],
                                            color=color, alpha=.1)
                        except KeyError:
                            print(' '.join(['failed in drawing the', content,'of', alg]))
                            continue
                    else:
                        try:
                            ax.plot(result['x'], floatmatrix(result['measure'][content])[:,0], color=color, label=sorted_alg_name, linewidth=width)
                            # ax.fill_between(result['x'], floatmatrix(result['measure'][content])[:,0] - floatmatrix(result['std'][content])[:,0],
                            #                 floatmatrix(result['measure'][content])[:,0] + floatmatrix(result['std'][content])[:,0],
                            #                 color=color, alpha=.1)
                        except KeyError:
                            print(' '.join(['failed in drawing the', content,'of', alg]))
                            continue


            color = COLORS_map['reference']
            if 'HalfCheetahEnv_cost' in env:
                ax.plot(result['x'], np.ones(result['measure'][content].shape[0]), color=color, label='reference',
                       linewidth=width, linestyle='dashed',)
            # if 'trunk_arm_sim' in env:
            #     ax.plot(result['x'], np.ones(result['measure'][content].shape[0]), color=color, label='reference',
            #            linewidth=width, linestyle='dashed',)

            if 'death_rate' in content:
                plt.xlim(80, 150)
            # if 'Half' in exp:
            #     plt.xlim(0.2, 2.)
            if 'ylim' in args.keys():
                plt.ylim(args['ylim'][0], args['ylim'][-1])
            ax.set_xlabel(result['x_name'])
            ax.set_ylabel(content)
            handles, labels = ax.get_legend_handles_labels()
            # item = handles.pop(1)
            # handles.insert(0,item)
            # item = labels.pop(1)
            # labels.insert(0, item)

            #ax.legend(handles, labels,fontsize=font, loc=2, fancybox=False, shadow=False)
            if eval_name=='constant_impulse':
                ax.legend([handles[2], handles[1], handles[0]], ['DeSKO', 'DKO', 'SAC'],
                          fontsize=font, loc=2, fancybox=False, shadow=False)
            elif 'trunk_arm_sim' in env:
                ax.legend([handles[2], handles[1], handles[0]], ['DeSKO', 'DKO', 'SAC'],
                          fontsize=font, loc=2, fancybox=False, shadow=False)

            else:
                ax.legend([handles[2], handles[1], handles[0], handles[3]], ['DeSKO', 'DKO', 'SAC', 'reference'],
                          fontsize=font, loc=2, fancybox=False, shadow=False)

            ax.grid(True)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

            if args['formal_plot']:
                plt.savefig(exp+'-' + eval_name + '-' + content + '.pdf')
                plt.show()

            else:
                ax.set_title(eval_name, fontsize=label_fontsize)
    if not args['formal_plot']:
        plt.show()

    return


def sorting_alg(algs):

    alg = list(range(3))
    index = 0
    for i in algs:
        if (i== 'Koopman_v10-jacob') or (i=='DeSKO') or (i== 'Koopman_v10-hmh'):


            alg[2]=i
        elif i == 'Koopman_v2-jacob' or (i=='DKO' )or (i== 'Koopman_v2-hmh'):
            alg[1]=i
        elif i == 'SAC_cost-jacob' or (i=='SAC_cost-jacob'):
            alg[0]=i
        else:
            alg[index]=i
        index += 1
    return alg




def plot_line_dynamic(env,content,results, alg_list, eval_contents, measure_list, exp, index, figsize = None):


    if len(eval_contents) == 0:
        return
    if args['formal_plot']:
        width = linewidth
        font = 20
    else:
        width = 1
        font = 14
    if not args['formal_plot']:
        nrows = len(measure_list)
        ncols = len(eval_contents)
        figsize = figsize or (18, 6)
        fig, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)
    # plot ep rewards
    for eval_index, eval_name in enumerate(eval_contents):

        for content_index, content in enumerate(measure_list):

            algs = list(results.keys())
            algs = sorting_alg(algs)
            if not args['formal_plot']:
                ax = axarr[content_index][eval_index]
            else:
                fig = plt.figure(figsize=(9, 6))
                ax = fig.add_subplot(111)
            for alg in algs:

                try:
                    result = results[alg][eval_name]
                except KeyError:

                    continue

                color_index = list(results.keys()).index(alg)
                sorted_alg_name = sort_alg_name(alg, args['formal_plot'])
                if args['formal_plot'] and sorted_alg_name in COLORS_map.keys():
                    color = COLORS_map[sorted_alg_name]
                else:
                    color = COLORS[color_index]

                lam = 0
                if content == 'lambda':
                    lam = 1
                if (lam == 0) & (alg != 'SAC_cost-jacob'):
                    try:
                        ax.plot(result['x'], floatmatrix(result['measure'][content])[:,index], color=color, label=sorted_alg_name, linewidth=width)
                        ax.fill_between(result['x'], floatmatrix(result['measure'][content])[:,index] - floatmatrix(result['std'][content])[:,index],
                                        floatmatrix(result['measure'][content])[:,index] + floatmatrix(result['std'][content])[:,index],
                                        color=color, alpha=.1)

                    except KeyError:
                        print(' '.join(['failed in drawing the', content,'of', alg]))
                        continue

                elif (lam != 0) & (alg != 'SAC_cost-jacob'):
                    try:
                        ax.plot(result['x'], floatmatrix(result['measure'][content])[:, index], color=color, label=sorted_alg_name,
                                linewidth=width)
                        ax.fill_between(result['x'],
                                        floatmatrix(result['measure'][content])[:, index] - lambdamatrix(result['std'][content]),
                                        floatmatrix(result['measure'][content])[:, index] + lambdamatrix(result['std'][content]),
                                        color=color, alpha=.1)
                    except KeyError:
                        print(' '.join(['failed in drawing the', content,'of', alg]))
                        continue

                elif ('oscillator' in env) or ('oscillator_observation_noise' in env) or ('oscillator_process_noise' in env):


                    try:
                        ax.plot(result['x'], result['measure'][content], color=color,
                                label=sorted_alg_name, linewidth=width)
                        ax.fill_between(result['x'], np.array(result['measure'][content] - result['std'][content],dtype=float),
                                        np.array(result['measure'][content] + result['std'][content],dtype=float),
                                        color=color, alpha=.1)
                    except KeyError:
                        print(' '.join(['failed in drawing the', content, 'of', alg]))
                        continue
                else:
                    try:
                        ax.plot(result['x'], floatmatrix(result['measure'][content])[:, index], color=color,
                                label=sorted_alg_name, linewidth=width)
                        ax.fill_between(result['x'], floatmatrix(result['measure'][content])[:, index] - floatmatrix(result['std'][content])[:, index],
                                        floatmatrix(result['measure'][content])[:, index] + floatmatrix(result['std'][content])[:, index],
                                        color=color, alpha=.1)
                    except KeyError:
                        print(' '.join(['failed in drawing the', content, 'of', alg]))
                        continue

            color = COLORS_map['reference']
            if ('oscillator' in env) or ('oscillator_observation_noise' in env) or ('oscillator_process_noise' in env):
                ax.plot(result['x'], floatmatrix(result['measure']['reference'])[:, index], color=color, label='reference',
                        linewidth=width, linestyle='dashed',)
            else:
                ax.plot(result['x'], floatmatrix(result['measure']['reference'])[:, index], color=color, label='reference',
                        linewidth=width, linestyle='dashed',)


            # ax.plot(result['x'], floatmatrix(result['measure'][content])[:, index], color=color, label='reference', linewidth = width, linestyle = 'dashed',)
            if 'death_rate' in content:
                plt.xlim(80, 150)
            # if 'Half' in exp:
            #     plt.xlim(0.2, 2.)
            if 'ylim' in args.keys():
                plt.ylim(args['ylim'][0], args['ylim'][-1])
            ax.set_xlabel(result['x_name'])
            ax.set_ylabel(content)
            handles, labels = ax.get_legend_handles_labels()
            # item = handles.pop(1)
            # handles.insert(0,item)
            # item = labels.pop(1)
            # labels.insert(0, item)

            #ax.legend(handles, labels,fontsize=font, loc=2, fancybox=False, shadow=False)
            ax.legend([handles[2], handles[1], handles[0], handles[3]], ['DeSKO', 'DKO', 'SAC', 'reference'], fontsize=font, loc=2, fancybox=False, shadow=False)
            ax.grid(True)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

            if args['formal_plot']:
                plt.savefig(exp+'-' + eval_name + '-' + content + '-%s.pdf' %index)
                plt.show()

            else:
                ax.set_title(eval_name, fontsize=label_fontsize)
    if not args['formal_plot']:
        plt.show()

    return

def main(args, alg_list, measure_list, env):
    results = load_results(args, alg_list, measure_list, env)
    if args['data'] == 'training':
        plot_training_results(results, alg_list, measure_list)
    else:
        line_plot_content = []
        mesh_plot_content = []
        for eval_content in args['eval_content']:
            if 'impulse' in eval_content or 'disturbance' in eval_content or 'dynamic' in eval_content:
                line_plot_content.append(eval_content)
            else:
                mesh_plot_content.append(eval_content)

        for exp in results.keys():
            plot_mesh(results[exp], alg_list, mesh_plot_content,measure_list)

            # if (args['eval_content'] == ['dynamic']) & (content != ['lambda']):
            if (args['eval_content'] == ['dynamic']) & (content != ['validation_loss'])& (env != ['HalfCheetahEnv_cost']) & (env != ['trunk_arm_sim']):
                for i in range(floatmatrix(results[env[0]][alg_list[0]][args['eval_content'][0]]['measure'][content[0]]).shape[1]):

                    plot_line_dynamic(env,content,results[exp], alg_list, line_plot_content, measure_list, exp, i)
            else:

                plot_line(env, results[exp], alg_list, line_plot_content,measure_list, exp)


    return

def sort_alg_name(alg, formal_plot):
    if formal_plot:
        if 'Koopman_v10' in alg:
            alg = 'DeSKO'
        elif 'Koopman_v2' in alg:
            alg = 'DKO'
        elif 'SAC_cost-jacob' in alg:
            alg = 'SAC'
        elif 'MLP-jacob' in alg:
            alg = 'MLP'
    return alg

if __name__ == '__main__':

    alg_list = [
        # 'LAC',
        # 'SAC_cost-new',
        # MJS1
        # 'LAC-lag',
        # 'Koopman_v5',
        # 'Koopman_v2',
        # 'Koopman_v10',
        # 'DKO',
        # 'DSKO_old',
        # 'DeSKO',
        'Koopman_v2-jacob',
        # 'MLP-jacob'
        # 'SAC_cost-jacob',
        'Koopman_v10-jacob',
        'MLP-jacob-new'
        # 'SAC_cost-jacob',
        # 'DeSKO',
        # 'Koopman_v10-bucket-256-lat=40',
        # 'Koopman_v2-new',
        # 'Koopman_v10-new',
        # 'DKO_old',
        # 'DSKO_old',
        # 'Koopman_v10-re-trial'
        # 'Koopman_v10-retry'
        # 'SAC',
        # 'SPPO',
        # 'LAC-relu',
        # 'LAC-biquad',
        # 'LAC-horizon=3-alpha3=.1',
        # osillator
        # 'LAC-horizon=5-a3=.1',
        # 'LAC-abs',
        # 'LAC-biquad',
        # 'LAC-quad',
        # 'LAC-lag-.0value-undiscount-lag',
        # 'LAC-lag-.01-value',
        # 'LAC-lag-.1-value',
        # osillator_complicated
        # 'LAC-horizon=5',

        # 'SDDPG',
        # 'LAC-horizon=inf-quadratic',
        # 'LAC-horizon=5-quadratic',
        # 'SAC',
        # 'N=5',
        # 'N=10',
        # 'N=15',
        # 'N=20',
        # 'infinite horizon',
        # 'LQR',
        # 'LAC-new',

        # 'LAC-horizon=10',
        # halfcheetah,
        # 'LAC-des=1-horizon=inf-alpha=1',

        #fetch
        # 'LAC',
        # 'SAC',
        ]

    args = {
        'data': ['training', 'eval'][0],
        'eval_content': [
            # 'length_of_pole-mass_of_cart',
            # 'impulse',
            'constant_impulse',
            # 'various_disturbance-sin',
            # 'mass_of_pole',
            # 'length_of_pole',
            # 'mass_of_cart',
            # 'dynamic',
        ],

        'plot_list': [str(i) for i in range(0, 5)],
        'formal_plot': True,
        # 'formal_plot': False,
        #  'ylim':[0,0.1],
        }
    # args = {'lim': [str(i) for i in range(0, 10)]}
    # args = {'plot_list': [str(i) for i in range(1)]}

    content = [
        # 'return',   ### constant impulse
        #  'average_path' ## dynamic
        # 'tracking_error',
        'validation_loss'
        # 'lambda',
        # 'eprewmean',
        # 'death_rate', ## dynamic (only for cartpole)
        # 'return_std',
        # 'average_length'
    ]
    env = [
        # 'cartpole_cost',
        # 'cartpole_random',
        # 'cartpole_observation_noise',
        # 'trunk_arm_sim'
        #  'HalfCheetahEnv_cost',
        # 'FetchReach-v1',
        # 'Antcost-v0',
        # 'Quadrotor-v1_cost',
        'oscillator',
        # 'oscillator_process_noise',
        # 'oscillator_observation_noise',
        # 'oscillator_complicated',
        # 'MJS1',
        # 'MJS2',
        ]#[8:9]

    main(args, alg_list, content, env)

