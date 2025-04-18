from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from log import Log
from loop_closure_gt import LoopClosureGt
import pandas as pd

@dataclass
class ConfusionMatrix():
    tp: int
    fp: int
    fn: int

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        return ConfusionMatrix(tp=self.tp+other.tp, fp=self.fp+other.fp, fn=self.fn+other.fn)

    def percision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

STAGE_NAMES = [
    'filter',
    'accumulate',
    'consistency',
    'map_point_matching',
    'ransac_pose_estimation',
    'pose_optimization',
    'reprojection'
]

def stage_is_binary(stage_name: str):
    return stage_name in ['accumulate', 'pose_optimization']

def compute_pr(kf: int, log: Log, loop_closure_gt: LoopClosureGt) -> Dict[str, ConfusionMatrix]:
    if kf not in log.initial_candidates:
        return {}

    kf_info = log.get_kf_info(kf)
    matched_kf = int(kf_info['matchedKf'])
    stage_candidates = [
        log.filtered_candidates.get(kf, []),
        log.acc_filtered_candidates.get(kf, []),
        log.consistent_candidates.get(kf, []),
        log.matched_frames.get(kf, []),
        log.ransac_solved_frames.get(kf, []),
        [matched_kf] if kf_info['poseOptimized'] > 0 else [],
        [matched_kf] if kf_info['computeSuccess'] > 0 else [],
    ]
    input_candidates = log.initial_candidates[kf]

    
    conf_dict = {}
    for output_candidates, name in zip(stage_candidates, STAGE_NAMES):
        if not input_candidates:
            conf_dict[name] = ConfusionMatrix(0, 0, 0)
            continue

        gt_candidates = loop_closure_gt.get_gt_candidates(kf, log.kf_ids(), log.connected_frames[kf], input_candidates)
        conf = ConfusionMatrix(
            tp=len(np.intersect1d(gt_candidates, output_candidates)),
            fp=len(np.setdiff1d(output_candidates, gt_candidates)),
            fn=len(np.setdiff1d(gt_candidates, output_candidates))
        )
        if stage_is_binary(name):
            # if name == 'pose_optimization':
            #     breakpoint()
            # If the stage can only output one frame, as long as we got one TP, we don't consider the other true candidates to be false negatives
            I = lambda x: int(x > 0)
            conf = ConfusionMatrix(tp=I(conf.tp), fp=I(conf.fp), fn=1-I(conf.tp))
        conf_dict[name] = conf
        input_candidates = output_candidates

    return conf_dict

if __name__ == '__main__':
    # 832
    log = Log('20250418_165109')
    # log = Log('20250418_174140')
    conf_dict_acc = {name: ConfusionMatrix(0,0,0) for name in STAGE_NAMES}
    for kf in log.kf_ids():
        conf_dict = compute_pr(kf, log, LoopClosureGt.for_kitti('06'))
        for name, conf in conf_dict.items():
            conf_dict_acc[name] += conf
    
    data = [
        {"stage": k, "percision": v.percision(), "recall": v.recall()}
        for k, v in conf_dict_acc.items()
    ]

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
