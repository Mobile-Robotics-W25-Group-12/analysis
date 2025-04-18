

from log import Log
from loop_closure_gt import LoopClosureGt


if __name__ == '__main__':
    log = Log("20250417_221714")
    # print(log.get_closed_points())
    # print(log.get_consistent_points())
    # print(log.kf_info)
    loop_closure_gt = LoopClosureGt.for_kitti('06')
    for kf in log.kf_ids():
        if kf not in log.connected_frames:
            continue
    
        candidates = loop_closure_gt.get_gt_candidates(kf, log.kf_ids(), log.connected_frames[kf][0])
        if candidates.shape[0] > 0:
            print(kf, candidates)

    breakpoint()