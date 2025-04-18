from pathlib import Path
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from matplotlib import pyplot as plt
from evo.core.trajectory import PoseTrajectory3D
import numpy as np
import copy

from evo.core import metrics
from evo.core.units import Unit

from log import Log

class TrajectoryAnalyzer():
    def __init__(self, est_file, ref_file, times_file):
        traj_ref = file_interface.read_kitti_poses_file(ref_file)
        traj_est = file_interface.read_kitti_poses_file(est_file)

        times = np.array([float(x) for x in Path(times_file).read_text().split()])

        traj_ref = PoseTrajectory3D(
            timestamps=times,
            poses_se3=traj_ref.poses_se3
        )

        traj_est = PoseTrajectory3D(
            timestamps=times,
            poses_se3=traj_est.poses_se3
        )

        max_diff = 0.01
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)

        self.traj_ref = traj_ref
        self.traj_est_aligned = traj_est_aligned

    @classmethod
    def for_kitti(cls, est_file: str, sequence_str: str):
        ref_file = f"kitti/poses/{sequence_str}.txt"
        times_file = f"kitti/times/{sequence_str}.txt"
        return cls(est_file, ref_file, times_file)
    
    def plot_trajectories(self):
        fig = plt.figure()
        traj_by_label = {
            "estimate (aligned)": self.traj_est_aligned,
            "reference": self.traj_ref
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()

    def get_ape_stats(self):
        pose_relation = metrics.PoseRelation.translation_part
        ape_metric = metrics.APE(pose_relation)
        ape_metric.process_data((self.traj_ref, self.traj_est_aligned))
        return ape_metric.get_all_statistics()
    
    def get_rpe_stats(self):
        pose_relation = metrics.PoseRelation.translation_part
        rpe_metric = metrics.RPE(pose_relation)
        rpe_metric.process_data((self.traj_ref, self.traj_est_aligned))
        return rpe_metric.get_all_statistics()

if __name__ == '__main__':
    log = Log("20250416_142550")
    analyzer = TrajectoryAnalyzer.for_kitti(log.kitti_trajectory_path, '06')
    # analyzer.plot_trajectories()
    print(analyzer.get_rpe_stats())