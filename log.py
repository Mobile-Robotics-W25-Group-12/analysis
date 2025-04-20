from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
import pandas
import yaml
import csv

load_dotenv()

ORB_SLAM_PATH = Path(os.getenv("ORB_SLAM_PATH"))
LOGS_PATH = ORB_SLAM_PATH / "logs"

class Log:
    def __init__(self, log_dir: Path, vectorFilepath: str = None):
        log_path = log_dir / "log.csv"
        self.log_df = pandas.read_csv(log_path)

        self.kf_info = {
            row["id"]: {key: row[key] for key in row.keys() if key != "id"}
            for _, row in self.log_df.iterrows()
        }

        with (log_dir / "params.yaml").open() as f:
            self.params = yaml.safe_load(f)

        self.vectors = None
        if "vectorFilepath" in self.params:
            self.vectors = np.load(ORB_SLAM_PATH / self.params["vectorFilepath"])
        if self.vectors is None and vectorFilepath is not None:
            self.vectors = np.load(vectorFilepath)

        def load_candidates(file_path):
            candidates_dict = {}
            with file_path.open() as f:
                reader = csv.reader(f)
                for row in reader:
                    frame_id = int(row[0])
                    candidates = list(map(int, row[1:]))
                    candidates_dict[frame_id] = candidates
            return candidates_dict

        def load_scored_candidates(file_path):
            candidates_dict = {}
            scores_dict = {}
            with file_path.open() as f:
                reader = csv.reader(f)
                for row in reader:
                    frame_id = int(row[0])
                    candidates = list(map(int, row[1::2]))
                    scores = list(map(float, row[2::2]))
                    candidates_dict[frame_id] = candidates
                    scores_dict[frame_id] = scores
            return candidates_dict, scores_dict

        self.connected_frames = load_candidates(log_dir / "connected_frames.csv")
        self.initial_candidates, _ = load_scored_candidates(log_dir / "initial_candidates.csv")
        self.filtered_candidates, _ = load_scored_candidates(log_dir / "filtered_candidates.csv")
        self.acc_filtered_candidates = load_candidates(log_dir / 'acc_filtered_candidates.csv')
        self.consistent_candidates, _ = load_scored_candidates(log_dir / "consistent_candidates.csv")
        self.matched_frames = load_candidates(log_dir / "matched_frames.csv")
        self.ransac_solved_frames = load_candidates(log_dir / "ransac_solved_frames.csv")

        self.kitti_trajectory_path = log_dir / "CameraTrajectoryKITTI.txt"

    @classmethod
    def from_logs_dir(cls, name: str):
        return cls(LOGS_PATH / name)

    def kf_ids(self):
        return self.log_df["id"].tolist()

    def get_kf_info(self, kf_id: int):
        return self.kf_info[kf_id]

    def get_consistent_points(self):
        df = self.log_df
        consistent = df[df["numConsistentCandidates"] > 0]
        return list(zip(consistent["id"], consistent["numConsistentCandidates"]))

    def get_closed_points(self):
        df = self.log_df
        closed = df[df["computeSuccess"] == 1]
        return list(zip(closed["id"], closed["matchedKf"]))


if __name__ == "__main__":
    log = Log.from_logs_dir("20250419_150027")
    print(log.get_closed_points())
    # print(log.get_consistent_points())
    # print(log.kf_info)
