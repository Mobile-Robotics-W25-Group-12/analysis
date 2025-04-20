from typing import List
import numpy as np

from log import Log


class LoopClosureGt():
    def __init__(self, filename: str):
        self.data = np.load(filename)
        # self.closure_frames = {}
        # for i in range(self.data.shape[0]):

    def get_gt_candidates(self, frame_id, keyframes: List[int], connected_keyframes: List[int], input_candidates: List[int] = None):
        col = self.data[:,frame_id]
        frames = np.where(col > 50)
        # breakpoint()
        frames = np.intersect1d(frames, keyframes)
        if input_candidates is not None:
            frames = np.intersect1d(frames, input_candidates)
        frames = frames[frame_id - frames > 100]
        return np.setdiff1d(frames, np.concatenate((connected_keyframes, [frame_id])))
    
    @classmethod
    def for_kitti(cls, sequence_str: str):
        return cls(f'kitti/scene_graphs/{sequence_str}.npy')

if __name__ == '__main__':
    # log = Log("20250418_165109")
    loop_closure_gt = LoopClosureGt.for_kitti('05')
    # for kf in log.kf_ids():
    #     if kf not in log.connected_frames:
    #         continue
    
    #     candidates = loop_closure_gt.get_gt_candidates(kf, np.arange(1100), [])
    #     if candidates.shape[0] > 0:
    #         print(kf, candidates)
    n_frames = 2761
    for kf in np.arange(n_frames):
        candidates = loop_closure_gt.get_gt_candidates(kf, np.arange(n_frames), [])
        if candidates.shape[0] > 0:
            print(kf, candidates)