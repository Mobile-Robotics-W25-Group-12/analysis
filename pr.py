from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from log import Log
from loop_closure_gt import LoopClosureGt

@dataclass
class ConfusionMatrix():
    tp: int
    fp: int
    fn: int

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        return ConfusionMatrix(tp=self.tp+other.tp, fp=self.fp+other.fp, fn=self.fn+other.fn)

STAGE_NAMES = [
    'filter',
    'consistency',
    'map_point_matching',
    'ransac_pose_estimation',
    'pose_optimization',
    'reprojection'
]

def compute_pr(kf: int, log: Log, loop_closure_gt: LoopClosureGt):
    if kf not in log.initial_candidates:
        return {}

    kf_info = log.get_kf_info(kf)
    stage_candidates = [
        log.filtered_candidates.get(kf, ([],))[0],
        log.consistent_candidates.get(kf, ([],))[0],
        log.matched_frames.get(kf, ([],))[0],
        log.ransac_solved_frames.get(kf, ([],))[0],
        [kf_info['matchedKf']] if kf_info['poseOptimized'] > 0 else [],
        [kf_info['matchedKf']] if kf_info['computeSuccess'] > 0 else [],
    ]
    
    input_candidates = log.initial_candidates[kf][0]
    
    stats = {}
    for output_candidates, name in zip(stage_candidates, STAGE_NAMES):
        gt_candidates = loop_closure_gt.get_gt_candidates(kf, log.kf_ids(), log.connected_frames[kf][0], input_candidates)
        stats[name] = ConfusionMatrix(
            tp=len(np.intersect1d(gt_candidates, output_candidates)),
            fp=len(np.setdiff1d(output_candidates, gt_candidates)),
            fn=len(np.setdiff1d(gt_candidates, output_candidates))
        )
        input_candidates = output_candidates

    return stats

if __name__ == '__main__':
    # 832
    log = Log('20250417_212733')
    conf_dict_acc = {name: ConfusionMatrix(0,0,0) for name in STAGE_NAMES}
    for kf in log.kf_ids():
        conf_dict = compute_pr(kf, log, LoopClosureGt.for_kitti('06'))
        for name, conf in conf_dict.items():
            conf_dict_acc[name] += conf
    print(conf_dict_acc)