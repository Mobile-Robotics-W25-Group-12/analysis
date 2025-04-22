# Analysis

This repository is for analyzing loop closure experiments ran in the ORB-SLAM2 repository.

## Setup

Create a `.env` file in this directory with `ORB_SLAM_PATH=path_to_your_orb_slam_2_repo`. This repository should contain your experiment folders in a directory `experiments`.

Install the requirements with `pip install -r requirements.txt`. Python 3.10+ required.

## Usage

### Running all analysis

In `main.py`, modify the example in the `main()` function to load your experiment(s) and then perform analysis. The script will output plots in the `plots` directory. These include a scatter plot of the loop closure times, a box-and-whisker plot of the absolute pose error RMSE, and a LaTeX formatted table of the precision and recall of each loop closure stage.

### Log file columns

Each experiment trial has a `log.csv` file, which is loaded into a pandas dataframe on the `Log` object for analysis. The columns are as follows.

- `numInitialCandidates` - number of candidates initially pulled from the database (this excludes connected frames and frames that share no words)
- `numFilteredCandidates` - number of filtered candidates (score greater than the lower score)
- `numAccFilteredCandidates` - number of accumulated candidates (candidates in the same covisibility group are aggregated into the candidate with the highest score)
- `numConsistentCandidates` - number of consistent candidates (their covisibility groups have non-zero overlap with candidate covisibility groups from the previous few frames [3 frames])
- `loopDetected` - `DetectLoop` returned true (iff there is at least one consisten candidate)
- `numMatchedFrames` - number of candidates that have enough initial map point matches (`numInitialMatchPoints`) with the current keyframe. Each of these matched frames gets a sim3 RANSAC solver constructed.
- `ransacPoseEstimateSolved` - RANSAC estimated the pose for at least one candidate keyframe. Requires at least `numRansacInliers` inliers. 
- `poseOptimized` - the sim3 output from at least one RANSAC was successfully optimized. Requires at least `numOptimizationInliers` inliers after optimization.
- `matchedKf` - the keyframe for which we have optimized the RANSAC pose
- `computeSuccess` - `ComputeSim3` returns true. Happens iff `poseOptimized` was true AND the reprojected matched keyframe had at least `numProjectedMatchPoints` map point matches with the current keyframe.
- `minScore` - the minimum BoW/BoQ score for filtering frames to get `numFilteredCandidates`
- `detectLoopDurationMs` - time to call `DetectLoop`
- `computeSimDurationMs` - time to call `ComputeSim3`
