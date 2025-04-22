from typing import List
import numpy as np

from log import Log


class LoopClosureGt:
    def __init__(self, filename: str):
        self.data = np.load(filename)

    def get_gt_candidates(
        self,
        frame_id,
        keyframes: List[int],
        connected_keyframes: List[int],
        input_candidates: List[int] = None,
    ):
        col = self.data[:, frame_id]
        frames = np.where(col > 50)
        frames = np.intersect1d(frames, keyframes)
        if input_candidates is not None:
            frames = np.intersect1d(frames, input_candidates)
        frames = frames[frame_id - frames > 100]
        return np.setdiff1d(frames, np.concatenate((connected_keyframes, [frame_id])))

    @classmethod
    def for_kitti(cls, sequence_str: str):
        return cls(f"kitti/scene_graphs/{sequence_str}.npy")
