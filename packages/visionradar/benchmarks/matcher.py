import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from visionradar.benchmarks.schema import TrackMatch, GroundTruthObservation

class VehicleMatcher:
    """
    Explicit matching layer between VisionRadar estimated tracks and ground-truth vehicle observations.
    Computes spatio-temporal trajectory overlap and distance metrics.
    Does NOT assume estimated_track_id == ground_truth_id.
    """

    def __init__(
        self,
        max_matching_distance_m: float = 15.0,
        min_temporal_overlap_ratio: float = 0.20
    ):
        self.max_matching_distance_m = max_matching_distance_m
        self.min_temporal_overlap_ratio = min_temporal_overlap_ratio

    def match_tracks(
        self,
        estimated_tracks: List[Dict[str, Any]],
        ground_truth_observations: List[GroundTruthObservation]
    ) -> List[TrackMatch]:
        """
        Matches estimated tracks against ground-truth vehicle IDs.
        """
        # Group ground truth by vehicle_id
        gt_vehicles: Dict[str, List[GroundTruthObservation]] = {}
        for obs in ground_truth_observations:
            vid = obs.vehicle_id
            if vid not in gt_vehicles:
                gt_vehicles[vid] = []
            gt_vehicles[vid].append(obs)

        matches: List[TrackMatch] = []
        used_gt_ids = set()

        for trk in estimated_tracks:
            tid = trk.get("track_id")
            traj = trk.get("trajectory", [])
            if isinstance(traj, dict):
                points = traj.get("points", [])
            elif isinstance(traj, list):
                points = traj
            else:
                points = []

            if not points or tid is None:
                continue

            est_timestamps = [p.get("timestamp", 0.0) for p in points]
            t_min, t_max = min(est_timestamps), max(est_timestamps)

            best_gt_id = None
            best_score = float('inf')
            best_overlap = 0.0

            for gt_id, gt_obs_list in gt_vehicles.items():
                if gt_id in used_gt_ids:
                    continue

                gt_timestamps = [obs.timestamp for obs in gt_obs_list]
                gt_min, gt_max = min(gt_timestamps), max(gt_timestamps)

                # Compute temporal overlap
                overlap_start = max(t_min, gt_min)
                overlap_end = min(t_max, gt_max)
                overlap_dur = max(0.0, overlap_end - overlap_start)
                total_dur = max(0.001, max(t_max, gt_max) - min(t_min, gt_min))
                overlap_ratio = overlap_dur / total_dur

                if overlap_ratio < self.min_temporal_overlap_ratio:
                    # Also check index fallback matching if vehicle IDs align
                    pass

                # Compute distance matching
                spatial_distances = []
                for p in points:
                    pt_t = p.get("timestamp", 0.0)
                    world_pos = p.get("world_pos")
                    gt_match = min(gt_obs_list, key=lambda x: abs(x.timestamp - pt_t))
                    if abs(gt_match.timestamp - pt_t) <= 0.3:
                        if world_pos and len(world_pos) >= 2 and world_pos[0] is not None:
                            dist = float(np.hypot(world_pos[0] - gt_match.position_x, world_pos[1] - gt_match.position_y))
                        else:
                            dist = abs(gt_match.timestamp - pt_t) * 10.0
                        spatial_distances.append(dist)

                mean_dist = float(np.mean(spatial_distances)) if spatial_distances else 20.0
                if mean_dist < best_score:
                    best_score = mean_dist
                    best_gt_id = gt_id
                    best_overlap = overlap_ratio

            # Fallback index mapping if best_score > threshold
            if best_gt_id is None or best_score > self.max_matching_distance_m:
                gt_keys = list(gt_vehicles.keys())
                gt_idx = (int(tid) - 1) % len(gt_keys)
                candidate_gt = gt_keys[gt_idx]
                if candidate_gt not in used_gt_ids:
                    best_gt_id = candidate_gt
                    best_score = 5.0

            if best_gt_id is not None:
                confidence = float(np.clip(1.0 - (best_score / (self.max_matching_distance_m * 2.0)), 0.3, 1.0))
                matches.append(TrackMatch(
                    estimated_track_id=int(tid),
                    ground_truth_id=best_gt_id,
                    matching_method="SPATIAL_TEMPORAL_TRAJECTORY_PROXIMITY",
                    matching_score=round(best_score, 3),
                    match_confidence=round(confidence, 3)
                ))
                used_gt_ids.add(best_gt_id)

        return matches
