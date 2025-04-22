import itertools
import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from log import Log
from loop_closure_gt import LoopClosureGt
from pr import compute_log_precision_recall
import seaborn as sns

from trajectory import TrajectoryAnalyzer

load_dotenv()

ORB_SLAM_PATH = Path(os.getenv("ORB_SLAM_PATH"))
EXPERIMENTS_PATH = ORB_SLAM_PATH / "experiments"

def get_average_pr(logs: List[Log], loop_closure_gt: LoopClosureGt) -> pd.DataFrame:
    dfs = [compute_log_precision_recall(log, loop_closure_gt) for log in logs]
    combined_df = pd.concat(dfs)
    averaged_df = combined_df.groupby('stage', as_index=False, sort=False).mean()
    return averaged_df

def get_combined_pr(logs_list: List[List[Log]], labels: List[str], loop_closure_gt: LoopClosureGt):
    pr_list = []
    for logs, labels in zip(logs_list, labels):
        pr = get_average_pr(logs, loop_closure_gt)
        pr_list.append(pr)

    combined_pr = pd.concat(
        {
            label: pr.set_index('stage') for label, pr in zip(labels, pr_list)
        },
        axis=1
    ).transpose()

    combined_pr = combined_pr.loc[:, pr_list[0]['stage']]
    return combined_pr

def get_first_loop_closure_points(logs: List[Log]):
    return [log.get_closed_points()[0][0] for log in logs]

def get_all_loop_closure_points(logs: List[Log]):
    return list((kf, i) for log in logs for i, (kf, _) in enumerate(log.get_closed_points()))

def plot_jitter(logs_list: List[List[Log]], labels, times, title: str, pdf_name = None):  
    import matplotlib.pyplot as plt

    sns.set_theme(font_scale=3)
    
    plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
    plt.rc('text', usetex=True)

    times_list = []
    indices_list = []
    for logs in logs_list:
        points_and_indices = get_all_loop_closure_points(logs)
        points = [x[0] for x in points_and_indices]
        indices = [x[1] + 1 for x in points_and_indices]
        times_list.append(times[points])
        indices_list.append(indices)

    data = {
        'Model': list(itertools.chain.from_iterable([label] * len(vals) for vals, label in zip(times_list, labels))),
        'Points': np.concatenate(times_list),
        'Indices': np.concatenate(indices_list)
    }

    df = pd.DataFrame(data)

    extra_params = {'hue': 'Indices', 'palette': 'flare', 'dodge': True} if df['Indices'].nunique() > 1 else {}
    # extra_params = {'hue': 'Indices', 'palette': 'flare', 'dodge': True}

    plt.figure(figsize=(14, 10))
    g = sns.swarmplot(x='Points',  y='Model', data=df, size=10, alpha=0.7, **extra_params)
    
    if g.legend_:
        # g.legend_.set_title('Loop Closure Index')
        g.legend_.remove()

    plt.xlabel('Time (s)')
    plt.ylabel('')

    # plt.ylabel('Model')
    # plt.title(title)

    if pdf_name:
        Path('plots').mkdir(exist_ok=True)
        plt.savefig(f'plots/{pdf_name}', format='pdf', bbox_inches='tight')

    plt.show()

def get_ape_rmses(logs: List[Log], sequence_name: str):
    rmse_vals = []
    for log in logs:
        analyzer = TrajectoryAnalyzer.for_kitti(log.kitti_trajectory_path, sequence_name)
        stats = analyzer.get_ape_stats()
        rmse_vals.append(stats['rmse'])
    rmse_vals = np.array(rmse_vals)
    return rmse_vals

def plot_error(vals_list, labels, title, pdf_name=None):
    import matplotlib.pyplot as plt

    sns.set_theme(font_scale=3)
    
    plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
    plt.rc('text', usetex=True)
    
    data = {
        'Model': list(itertools.chain.from_iterable([label] * len(vals) for vals, label in zip(vals_list, labels))),
        'RMSE': np.concatenate(vals_list)
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 4))
    fig = sns.boxplot(y='Model', x='RMSE', data=df)

    plt.ylabel('')
    plt.xlabel('APE RMSE')
    # plt.title(title)
    
    if pdf_name:
        Path('plots').mkdir(exist_ok=True)
        plt.savefig(f'plots/{pdf_name}', format='pdf', bbox_inches='tight')

    plt.show()

def plot_trajectory(log: Log, sequence_name: str):
    analyzer = TrajectoryAnalyzer.for_kitti(log.kitti_trajectory_path, sequence_name)
    analyzer.plot_trajectories()

def load_experiment(experiment_name: str):
    experiment_dir = EXPERIMENTS_PATH / experiment_name

    logs_dict: Dict[str, List[Log]] = {}

    with (experiment_dir / 'test_cases.json').open() as f:
        test_cases = json.load(f)
    for test_case in test_cases:
        logs_dict[test_case] = []
        test_case_dir = experiment_dir / test_case
        with (test_case_dir / 'trials.json').open() as f:
            trials = json.load(f)
        for trial in trials:
            log = Log(test_case_dir / trial)
            logs_dict[test_case].append(log)
    return logs_dict

def analyze(logs_list: List[List[Log]], labels: List[str], sequence_name: str):
    rmses = []
    for logs, label in zip(logs_list, labels):
        rmses.append(get_ape_rmses(logs, sequence_name))
    # plot_trajectory(logs_dict['boq'][0], '06')

    plot_error(rmses, labels, f'Absolute Pose Error final RMSE in KITTI {sequence_name} Augmented Brightness', f'{sequence_name}_ape.pdf')

    times = np.array([float(x) for x in Path(f'kitti/times/{sequence_name}.txt').read_text().split()])
    plot_jitter(logs_list, labels, times, f'Time of loop closure in KITTI {sequence_name} Augmented Brightness', f'{sequence_name}_times.pdf')
    
    loop_closure_gt = LoopClosureGt.for_kitti(sequence_name)
    combined_pr = get_combined_pr(logs_list, labels, loop_closure_gt)
    
    print(combined_pr.to_latex(float_format = '{:.2g}'.format))
    # print(combined_pr)

if __name__ == '__main__':
    sequence_name = '05'

    logs_dict_default = load_experiment(f'{sequence_name}_brightness')
    logs_dict_04 = load_experiment(f'{sequence_name}_brightness_0.4')
    logs_dict_no_consistency = load_experiment(f'{sequence_name}_brightness_0.4_no_consistency')

    logs_list = [
        logs_dict_default['bow'],
        logs_dict_no_consistency['bow'],
        logs_dict_default['boq'],
        logs_dict_04['boq'],
        logs_dict_no_consistency['boq'],
    ]

    labels = ['BoW [c=3]', 'BoW [c=1]', 'BoQ [m=0.6, c=3]', 'BoQ [m=0.4, c=3]', 'BoQ [m=0.4, c=1]']

    analyze(logs_list, labels, sequence_name)
