from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from log import Log
from loop_closure_gt import LoopClosureGt
import pandas as pd


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )

    def precision(self):
        # if self.tp == 0 and self.fp == 0 and self.fn == 0:
        #     return float('NaN')
        if self.tp == 0 and self.fp == 0:
            return 1
        return self.tp / (self.tp + self.fp)

    def recall(self):
        # if self.tp == 0 and self.fp == 0 and self.fn == 0:
        #     return float('NaN')
        if self.tp == 0 and self.fn == 0:
            return 1
        return self.tp / (self.tp + self.fn)


STAGE_NAMES = [
    "filter",
    "accumulate",
    "consistency",
    "map_point_matching",
    "ransac_pose_estimation",
    "pose_optimization",
    "reprojection",
]


def stage_is_binary(stage_name: str):
    return stage_name in ["accumulate", "pose_optimization"]


def compute_confusion(
    kf: int, log: Log, loop_closure_gt: LoopClosureGt
) -> Dict[str, ConfusionMatrix]:
    if kf not in log.initial_candidates:
        return {}

    kf_info = log.get_kf_info(kf)
    matched_kf = int(kf_info["matchedKf"])
    stage_candidates = [
        log.filtered_candidates.get(kf, []),
        log.acc_filtered_candidates.get(kf, []),
        log.consistent_candidates.get(kf, []),
        log.matched_frames.get(kf, []),
        log.ransac_solved_frames.get(kf, []),
        [matched_kf] if kf_info["poseOptimized"] > 0 else [],
        [matched_kf] if kf_info["computeSuccess"] > 0 else [],
    ]
    input_candidates = log.initial_candidates[kf]

    conf_dict = {}
    for output_candidates, name in zip(stage_candidates, STAGE_NAMES):
        if not input_candidates:
            conf_dict[name] = ConfusionMatrix()
            continue

        gt_candidates = loop_closure_gt.get_gt_candidates(
            kf, log.kf_ids(), log.connected_frames[kf], input_candidates
        )
        conf = ConfusionMatrix(
            tp=len(np.intersect1d(gt_candidates, output_candidates)),
            fp=len(np.setdiff1d(output_candidates, gt_candidates)),
            fn=len(np.setdiff1d(gt_candidates, output_candidates)),
            tn=len(
                np.setdiff1d(
                    np.setdiff1d(input_candidates, gt_candidates), output_candidates
                )
            ),
        )
        if stage_is_binary(name):
            # If the stage can only output one frame, as long as we got one TP, we don't consider the other true candidates to be false negatives
            I = lambda x: int(x > 0)
            conf = ConfusionMatrix(
                tp=I(conf.tp),
                fp=I(conf.fp),
                fn=I(conf.tp == 0 and len(gt_candidates) > 0),
            )
        conf_dict[name] = conf
        input_candidates = output_candidates

    return conf_dict


def compute_log_precision_recall(
    log: Log, loop_closure_gt: LoopClosureGt
) -> pd.DataFrame:
    conf_dict_acc = {name: ConfusionMatrix() for name in STAGE_NAMES}
    for kf in log.kf_ids():
        conf_dict = compute_confusion(kf, log, loop_closure_gt)
        for name, conf in conf_dict.items():
            conf_dict_acc[name] += conf

    data = [
        {"stage": k, "precision": v.precision(), "recall": v.recall()}
        for k, v in conf_dict_acc.items()
    ]

    df = pd.DataFrame(data)
    return df
