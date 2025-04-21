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

def get_loop_closure_points(logs: List[Log]):
    return [log.get_closed_points()[0][0] for log in logs]

def plot_jitter(logs_dict: Dict[str, List[Log]], times, title: str):  
    import matplotlib.pyplot as plt

    sns.set_theme(font_scale=2)

    boq_points = get_loop_closure_points(logs_dict['boq'])
    bow_points = get_loop_closure_points(logs_dict['bow'])

    boq_times = times[boq_points]
    bow_times = times[bow_points]

    data = {
        'Model': ['BoQ'] * len(boq_times) + ['BoW'] * len(bow_times),
        'Points': np.concatenate((boq_times, bow_times))
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Points',  y='Model', data=df, size=16, alpha=0.7, 
        palette={'BoQ': 'green', 'BoW': 'blue'}
    )

    plt.xlabel('Time (s)')
    plt.ylabel('Model')
    plt.title(title)

    plt.show()

def get_ape_rmses(logs: List[Log], sequence_name: str):
    rmse_vals = []
    for log in logs:
        analyzer = TrajectoryAnalyzer.for_kitti(log.kitti_trajectory_path, sequence_name)
        stats = analyzer.get_ape_stats()
        rmse_vals.append(stats['rmse'])
    rmse_vals = np.array(rmse_vals)
    return rmse_vals

def plot_error(boq_vals, bow_vals, title):
    import matplotlib.pyplot as plt

    sns.set_theme(font_scale=2)
    
    data = {
        'Model': ['BoQ'] * len(boq_vals) + ['BoW'] * len(bow_vals),
        'RMSE': np.concatenate((boq_vals, bow_vals))
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(y='Model', x='RMSE', data=df, palette={'BoQ': 'green', 'BoW': 'blue'})

    plt.ylabel('Model')
    plt.xlabel('APE RMSE')
    plt.title(title)

    plt.show()

def plot_trajectory(log: Log, sequence_name: str):
    analyzer = TrajectoryAnalyzer.for_kitti(log.kitti_trajectory_path, sequence_name)
    analyzer.plot_trajectories()

def analyze(experiment_name, sequence_name):
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

    boq_rmse = get_ape_rmses(logs_dict['boq'], sequence_name)
    bow_rmse = get_ape_rmses(logs_dict['bow'], sequence_name)
    # print(boq_rmse)
    # plot_trajectory(logs_dict['boq'][0], '06')

    plot_error(boq_rmse, bow_rmse, 'Absolute Pose Error final RMSE in KITTI 06 Augmented Brightness')

    times = np.array([float(x) for x in Path(f'kitti/times/{sequence_name}.txt').read_text().split()])
    plot_jitter(logs_dict, times, 'Time of first loop closure in KITTI 06 Augmented Brightness')

    loop_closure_gt = LoopClosureGt.for_kitti(sequence_name)
    boq_pr = get_average_pr(logs_dict['boq'], loop_closure_gt)
    bow_pr = get_average_pr(logs_dict['bow'], loop_closure_gt)

    combined_pr = pd.concat(
        {
            'BoQ': boq_pr.set_index('stage'),
            'BoW': bow_pr.set_index('stage')
        },
        axis=1
    ).transpose()

    combined_pr = combined_pr.loc[:, boq_pr['stage']]
    # print(combined_pr.to_latex(float_format = '{:.2g}'.format))
    print(combined_pr)

if __name__ == '__main__':
    analyze('06_brightness', '06')
