
import numpy as np
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Set, Callable
from enum import Enum
import config

logger = logging.getLogger("VehicleIDCorrector")


class CorrectiveAction(Enum):
    MERGE_IDS = "merge_ids"
    REMOVE_DUPLICATE = "remove_duplicate"
    CLEAN_TRAJECTORY = "clean_trajectory"
    NO_ACTION = "no_action"


@dataclass
class VehicleTrajectoryRecord:
    global_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    cameras: deque = field(default_factory=lambda: deque(maxlen=50))
    lanes: deque = field(default_factory=lambda: deque(maxlen=50))
    last_update: float = 0.0
    creation_time: float = 0.0
    confidence_score: float = 1.0

    def add_position(self, position: Tuple[int, int], timestamp: float,
                     camera_id: str, lane_y: float):
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.cameras.append(camera_id)
        self.lanes.append(lane_y)
        self.last_update = timestamp

        if not hasattr(self, 'creation_time') or self.creation_time == 0.0:
            self.creation_time = timestamp

    def get_speed_estimate(self) -> float:
        if len(self.positions) < 2:
            return 0.0

        total_distance = 0.0
        total_time = 0.0

        for i in range(1, len(self.positions)):
            pos1, pos2 = self.positions[i - 1], self.positions[i]
            time1, time2 = self.timestamps[i - 1], self.timestamps[i]

            distance = float(np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2))
            time_diff = float(time2 - time1)

            if time_diff > 0:
                total_distance += distance
                total_time += time_diff

        return float(total_distance / total_time) if total_time > 0 else 0.0

    def get_movement_direction(self) -> Optional[str]:
        if len(self.positions) < 2:
            return None

        start_pos = self.positions[0]
        end_pos = self.positions[-1]

        dx = end_pos[0] - start_pos[0]

        if abs(dx) < 10:
            return "static"
        elif dx > 0:
            return "right"
        else:
            return "left"


@dataclass
class ConflictEvidence:
    evidence_type: str
    confidence: float
    description: str
    involved_ids: Set[int]
    timestamp: float


@dataclass
class CameraSwitchContext:
    global_id: int
    from_camera: str
    to_camera: str
    from_time: float
    to_time: float
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    from_lane: float
    to_lane: float
    from_dir: Optional[str]
    to_dir: Optional[str]
    track_key: Optional[Tuple[str, int]] = None
    reid_score: Optional[float] = None


@dataclass
class HandoverEvent:
    global_id: int
    camera_id: str
    timestamp: float
    position: Tuple[int, int]
    lane_y: float
    direction: Optional[str]
    event_type: str  # "entry" or "exit"


@dataclass
class CorrectionEvent:
    action: str  # "split_id"
    global_id: int
    camera_id: str
    reason: str
    track_key: Optional[Tuple[str, int]] = None
    score: Optional[float] = None
    timestamp: float = 0.0


class VehicleIDCorrector:

    def __init__(self, tunnel_map):
        self.tunnel_map = tunnel_map
        self.trajectory_records: Dict[int, VehicleTrajectoryRecord] = {}
        self.correction_history: List[Dict] = []

        self.correction_interval = 5.0
        self.trajectory_timeout = 30.0

        self.simultaneous_render_threshold = 2.0
        self.merge_confidence_threshold = 0.75
        self.rapid_reassignment_window = 10.0
        self.early_termination_threshold = 5.0

        self.evidence_weights = {
            'temporal': 0.30,
            'spatial': 0.25,
            'direction': 0.20,
            'speed': 0.15,
            'lane': 0.10,
        }

        self.camera_switch_min_time = getattr(config, "CAMERA_SWITCH_MIN_TIME", 0.5)
        self.camera_switch_max_time = getattr(config, "CAMERA_SWITCH_MAX_TIME", 20.0)
        self.camera_switch_score_threshold = getattr(config, "CAMERA_SWITCH_SCORE_THRESHOLD", 0.4)
        self.camera_switch_cooldown = getattr(config, "CAMERA_SWITCH_COOLDOWN", 3.0)
        self.max_camera_skip = getattr(config, "MAX_CAMERA_SKIP", 1)
        self.max_cross_camera_speed = getattr(config, "MAX_CROSS_CAMERA_SPEED", 300.0)

        self.last_seen_by_id: Dict[int, Tuple[str, float, Tuple[int, int], float]] = {}
        self.invalid_switch_cache: Dict[Tuple[int, str], float] = {}
        self.on_correction_callback: Optional[Callable[[CorrectionEvent], None]] = None
        self.last_correction_time = 0.0

        self.stats = {
            'total_corrections': 0,
            'successful_merges': 0,
            'trajectory_cleanups': 0,
            'rapid_reassignment_corrections': 0,
            'split_corrections': 0,
        }

        logger.info("车辆ID纠错器已初始化（重构版 - 统一证据评分）")

    # ================================================================
    # ================================================================

    def set_correction_callback(self, callback: Callable[[CorrectionEvent], None]):
        self.on_correction_callback = callback

    def update_vehicle_trajectory(self, global_id: int, position: Tuple[int, int],
                                  timestamp: float, camera_id: str, lane_y: float,
                                  local_id: Optional[int] = None,
                                  reid_score: Optional[float] = None) -> bool:
        record = self.trajectory_records.get(global_id)
        prev_info = self.last_seen_by_id.get(global_id)
        accept_update = True

        if prev_info and prev_info[0] != camera_id:
            from_camera, from_time, from_pos, from_lane = prev_info
            from_dir = record.get_movement_direction() if record else None
            to_dir = self._estimate_direction(from_pos, position)
            ctx = CameraSwitchContext(
                global_id=global_id,
                from_camera=from_camera,
                to_camera=camera_id,
                from_time=from_time,
                to_time=timestamp,
                from_pos=from_pos,
                to_pos=position,
                from_lane=from_lane,
                to_lane=lane_y,
                from_dir=from_dir,
                to_dir=to_dir,
                track_key=(camera_id, local_id) if local_id is not None else None,
                reid_score=reid_score
            )

            if not self._is_in_switch_cooldown(global_id, camera_id, timestamp):
                is_valid, score, reason = self._validate_camera_switch(ctx)
                if not is_valid:
                    accept_update = False
                    self._register_split_event(ctx, reason, score, timestamp)
            else:
                accept_update = False

        if global_id not in self.trajectory_records:
            self.trajectory_records[global_id] = VehicleTrajectoryRecord(
                global_id=global_id,
                creation_time=timestamp
            )

        if accept_update:
            record = self.trajectory_records[global_id]
            record.add_position(position, timestamp, camera_id, lane_y)
            self.last_seen_by_id[global_id] = (camera_id, timestamp, position, lane_y)

        if timestamp - self.last_correction_time >= self.correction_interval:
            self.execute_correction_cycle(timestamp)
            self.last_correction_time = timestamp

        return accept_update

    def execute_correction_cycle(self, current_time: float):
        try:
            logger.debug(f"开始执行纠错周期，当前时间: {current_time:.2f}")

            self._cleanup_expired_trajectories(current_time)

            conflicts = self._detect_all_conflicts(current_time)

            if conflicts:
                logger.info(f"检测到 {len(conflicts)} 个潜在冲突")

                for conflict in conflicts:
                    self._resolve_conflict(conflict, current_time)

        except Exception as e:
            logger.error(f"纠错周期执行出错: {e}")

    def get_correction_statistics(self) -> Dict:
        return {
            **self.stats,
            'active_trajectories': len(self.trajectory_records),
            'correction_history_count': len(self.correction_history),
            'last_correction_time': self.last_correction_time
        }

    def force_correction_check(self, current_time: float):
        logger.info("强制执行纠错检查")
        self.execute_correction_cycle(current_time)

    # ================================================================
    # ================================================================

    def _detect_all_conflicts(self, current_time: float) -> List[Tuple[int, int, float]]:
        conflicts: List[Tuple[int, int, float]] = []
        seen_pairs: Set[Tuple[int, int]] = set()

        active_ids: List[int] = []
        recently_inactive_ids: List[int] = []

        for gid, record in self.trajectory_records.items():
            time_since = current_time - record.last_update
            duration = record.last_update - record.creation_time

            if time_since <= self.simultaneous_render_threshold:
                active_ids.append(gid)
            elif (time_since <= self.rapid_reassignment_window
                  and duration <= self.early_termination_threshold):
                recently_inactive_ids.append(gid)

        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id1, id2 = active_ids[i], active_ids[j]
                pair = (min(id1, id2), max(id1, id2))
                if pair in seen_pairs:
                    continue

                confidence = self._compute_evidence_score(
                    self.trajectory_records[id1],
                    self.trajectory_records[id2],
                    current_time,
                )
                if confidence >= self.merge_confidence_threshold:
                    conflicts.append((id1, id2, confidence))
                    seen_pairs.add(pair)

        for active_id in active_ids:
            for inactive_id in recently_inactive_ids:
                pair = (min(active_id, inactive_id), max(active_id, inactive_id))
                if pair in seen_pairs:
                    continue

                active_rec = self.trajectory_records[active_id]
                inactive_rec = self.trajectory_records[inactive_id]

                if not self._passes_rapid_reassignment_precheck(active_rec, inactive_rec):
                    continue

                confidence = self._compute_evidence_score(
                    inactive_rec, active_rec, current_time
                )
                if confidence >= self.merge_confidence_threshold:
                    conflicts.append((inactive_id, active_id, confidence))
                    seen_pairs.add(pair)

        return conflicts

    def _compute_evidence_score(self, record1: VehicleTrajectoryRecord,
                                record2: VehicleTrajectoryRecord,
                                current_time: float) -> float:
        scores: Dict[str, float] = {}

        scores['temporal'] = self._score_temporal_continuity(record1, record2)
        scores['spatial'] = self._score_spatial_proximity(record1, record2)
        scores['direction'] = self._score_direction_consistency(record1, record2)
        scores['speed'] = self._score_speed_consistency(record1, record2)
        scores['lane'] = self._score_lane_consistency(record1, record2)

        total = sum(scores[k] * self.evidence_weights[k] for k in self.evidence_weights)
        return min(1.0, total)

    def _score_temporal_continuity(self, r1: VehicleTrajectoryRecord,
                                   r2: VehicleTrajectoryRecord) -> float:
        if not r1.timestamps or not r2.timestamps:
            return 0.0

        end1, start2 = max(r1.timestamps), min(r2.timestamps)
        end2, start1 = max(r2.timestamps), min(r1.timestamps)

        gap1to2 = start2 - end1
        gap2to1 = start1 - end2
        min_gap = min(abs(gap1to2), abs(gap2to1))

        if min_gap <= 0.5:
            return 0.9
        elif min_gap <= 1.0:
            return 1.0
        elif min_gap <= 3.0:
            return 0.8
        elif min_gap <= 5.0:
            return 0.6
        elif min_gap <= 10.0:
            return 0.4
        elif min_gap <= 15.0:
            return 0.2
        else:
            return 0.0

    def _score_spatial_proximity(self, r1: VehicleTrajectoryRecord,
                                 r2: VehicleTrajectoryRecord) -> float:
        if not r1.positions or not r2.positions:
            return 0.0

        pairs = []
        if r1.positions and r2.positions:
            pairs.append((r1.positions[-1], r2.positions[0]))
            pairs.append((r2.positions[-1], r1.positions[0]))

        best_score = 0.0
        for p1, p2 in pairs:
            dist = float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
            if dist <= 20:
                score = 1.0
            elif dist <= 50:
                score = 0.8
            elif dist <= 100:
                score = 0.5
            elif dist <= 150:
                score = 0.2
            else:
                score = 0.0
            best_score = max(best_score, score)

        return best_score

    def _score_direction_consistency(self, r1: VehicleTrajectoryRecord,
                                     r2: VehicleTrajectoryRecord) -> float:
        dir1 = r1.get_movement_direction()
        dir2 = r2.get_movement_direction()

        if dir1 is None or dir2 is None:
            return 0.5
        if dir1 == dir2:
            return 1.0
        if dir1 == "static" or dir2 == "static":
            return 0.6
        return 0.0

    def _score_speed_consistency(self, r1: VehicleTrajectoryRecord,
                                 r2: VehicleTrajectoryRecord) -> float:
        speed1 = r1.get_speed_estimate()
        speed2 = r2.get_speed_estimate()

        if speed1 <= 0 or speed2 <= 0:
            return 0.5

        ratio = min(speed1, speed2) / max(speed1, speed2)
        return ratio

    def _score_lane_consistency(self, r1: VehicleTrajectoryRecord,
                                r2: VehicleTrajectoryRecord) -> float:
        if not r1.lanes or not r2.lanes:
            return 0.5

        avg_lane1 = float(np.mean(list(r1.lanes)))
        avg_lane2 = float(np.mean(list(r2.lanes)))
        diff = abs(avg_lane1 - avg_lane2)

        if diff <= 10:
            return 1.0
        elif diff <= 30:
            return 1.0 - (diff - 10) / 20.0
        else:
            return 0.0

    def _passes_rapid_reassignment_precheck(self, active_record: VehicleTrajectoryRecord,
                                            inactive_record: VehicleTrajectoryRecord) -> bool:
        inactive_end = inactive_record.last_update
        active_start = active_record.creation_time
        time_gap = active_start - inactive_end

        if not (-2.0 <= time_gap <= 5.0):
            return False

        if (inactive_record.positions and active_record.positions
                and len(inactive_record.positions) > 0 and len(active_record.positions) > 0):
            last_pos = inactive_record.positions[-1]
            first_pos = active_record.positions[0]
            distance = float(np.sqrt(
                (last_pos[0] - first_pos[0]) ** 2 + (last_pos[1] - first_pos[1]) ** 2
            ))
            if distance > 100:
                return False

        if (inactive_record.cameras and active_record.cameras
                and len(inactive_record.cameras) > 0 and len(active_record.cameras) > 0):
            last_cam = inactive_record.cameras[-1]
            first_cam = active_record.cameras[0]
            if not self._is_valid_camera_transition(last_cam, first_cam):
                return False

        return True

    # ================================================================
    # ================================================================

    def _resolve_conflict(self, conflict: Tuple[int, int, float], current_time: float):
        id1, id2, confidence = conflict

        try:
            record1 = self.trajectory_records.get(id1)
            record2 = self.trajectory_records.get(id2)

            if not record1 or not record2:
                return

            duration1 = record1.last_update - record1.creation_time
            duration2 = record2.last_update - record2.creation_time
            is_rapid_reassignment = (
                duration1 <= self.early_termination_threshold
                or duration2 <= self.early_termination_threshold
            )

            if record1.creation_time <= record2.creation_time:
                keep_id, merge_id = id1, id2
                keep_record, merge_record = record1, record2
            else:
                keep_id, merge_id = id2, id1
                keep_record, merge_record = record2, record1

            self._merge_trajectory_data(keep_record, merge_record)
            self._update_tunnel_map_for_merge(keep_id, merge_id)
            self._purge_id_state(merge_id)

            action_type = "快速重分配纠错" if is_rapid_reassignment else "同时渲染纠错"
            self.correction_history.append({
                'action': CorrectiveAction.MERGE_IDS.value,
                'timestamp': current_time,
                'kept_id': keep_id,
                'merged_id': merge_id,
                'confidence': confidence,
                'description': f"{action_type}: 合并ID {merge_id} 到 {keep_id}",
                'is_rapid_reassignment': is_rapid_reassignment
            })

            self.stats['total_corrections'] += 1
            self.stats['successful_merges'] += 1
            if is_rapid_reassignment:
                self.stats['rapid_reassignment_corrections'] += 1

            logger.info(f"成功{action_type}: {merge_id} -> {keep_id}，置信度: {confidence:.3f}")

        except Exception as e:
            logger.error(f"冲突解决失败: {e}")

    def _merge_trajectory_data(self, keep_record: VehicleTrajectoryRecord,
                               merge_record: VehicleTrajectoryRecord):
        all_points = []
        for pos, ts, cam, lane in zip(
            keep_record.positions, keep_record.timestamps,
            keep_record.cameras, keep_record.lanes
        ):
            all_points.append((ts, pos, cam, lane))
        for pos, ts, cam, lane in zip(
            merge_record.positions, merge_record.timestamps,
            merge_record.cameras, merge_record.lanes
        ):
            all_points.append((ts, pos, cam, lane))

        all_points.sort(key=lambda x: x[0])

        keep_record.positions.clear()
        keep_record.timestamps.clear()
        keep_record.cameras.clear()
        keep_record.lanes.clear()

        for ts, pos, cam, lane in all_points:
            keep_record.positions.append(pos)
            keep_record.timestamps.append(ts)
            keep_record.cameras.append(cam)
            keep_record.lanes.append(lane)

        if all_points:
            keep_record.creation_time = min(keep_record.creation_time, merge_record.creation_time)
            keep_record.last_update = all_points[-1][0]

        keep_record.confidence_score = max(
            keep_record.confidence_score, merge_record.confidence_score
        )

    # ================================================================
    # ================================================================

    def _validate_camera_switch(self, ctx: CameraSwitchContext) -> Tuple[bool, float, str]:
        if ctx.from_camera == ctx.to_camera:
            return True, 1.0, "同一摄像头"

        if not self._is_valid_camera_transition(ctx.from_camera, ctx.to_camera):
            return False, 0.0, "拓扑不允许的摄像头切换"

        time_gap = ctx.to_time - ctx.from_time
        if time_gap <= 0:
            return False, 0.0, f"时间顺序异常: {time_gap:.2f}s"

        min_time, max_time = self._get_transition_time_window(ctx.from_camera, ctx.to_camera)
        if time_gap < min_time or time_gap > max_time:
            return False, 0.0, f"时间窗不匹配: {time_gap:.2f}s 不在 [{min_time}, {max_time}]s"

        speed_kmh = self._estimate_cross_camera_speed_kmh(ctx.from_pos, ctx.to_pos, time_gap)
        if speed_kmh is not None and speed_kmh > self.max_cross_camera_speed:
            return False, 0.0, (
                f"跨摄像头速度异常: {speed_kmh:.1f}km/h > {self.max_cross_camera_speed:.1f}km/h"
            )

        direction_score = 0.5
        if ctx.from_dir and ctx.to_dir:
            if ctx.from_dir == ctx.to_dir:
                direction_score = 1.0
            elif ctx.from_dir == "static" or ctx.to_dir == "static":
                direction_score = 0.6
            else:
                direction_score = 0.0

        mid_time = (min_time + max_time) / 2.0
        temporal_score = 1.0 - min(
            abs(time_gap - mid_time) / max(max_time - min_time, 1e-6), 1.0
        )

        skip_count = self._get_camera_skip_count(ctx.from_camera, ctx.to_camera)
        skip_penalty = 0.1 * max(skip_count, 0)

        score = 0.7 * temporal_score + 0.3 * direction_score
        score = max(0.0, score - skip_penalty)

        threshold = self.camera_switch_score_threshold + 0.1 * max(skip_count, 0)

        if score >= threshold:
            return True, score, "切换评分通过"

        return False, score, f"切换评分不足({score:.3f} < {threshold:.3f})"

    def _register_split_event(self, ctx: CameraSwitchContext, reason: str,
                              score: Optional[float], timestamp: float):
        self.invalid_switch_cache[(ctx.global_id, ctx.to_camera)] = timestamp
        event = CorrectionEvent(
            action="split_id",
            global_id=ctx.global_id,
            camera_id=ctx.to_camera,
            reason=reason,
            track_key=ctx.track_key,
            score=score,
            timestamp=timestamp
        )
        self.stats['total_corrections'] += 1
        self.stats['split_corrections'] += 1
        logger.warning(
            f"检测到跨摄像头异常切换: ID {ctx.global_id} {ctx.from_camera} -> "
            f"{ctx.to_camera}, 原因: {reason}, 评分: {score if score is not None else -1:.3f}"
        )
        self._emit_correction_event(event)

    def _emit_correction_event(self, event: CorrectionEvent):
        self.correction_history.append({
            'action': event.action,
            'timestamp': event.timestamp,
            'global_id': event.global_id,
            'camera_id': event.camera_id,
            'reason': event.reason,
            'score': event.score,
            'track_key': event.track_key
        })

        if self.on_correction_callback is not None:
            try:
                self.on_correction_callback(event)
            except Exception as e:
                logger.error(f"纠错回调执行失败: {e}")

    def _is_in_switch_cooldown(self, global_id: int, camera_id: str,
                               current_time: float) -> bool:
        key = (global_id, camera_id)
        last_time = self.invalid_switch_cache.get(key)
        if last_time is None:
            return False
        return current_time - last_time <= self.camera_switch_cooldown

    # ================================================================
    # ================================================================

    def _cleanup_expired_trajectories(self, current_time: float):
        expired_ids = [
            gid for gid, record in self.trajectory_records.items()
            if current_time - record.last_update > self.trajectory_timeout
        ]

        for gid in expired_ids:
            self._purge_id_state(gid)
            self.stats['trajectory_cleanups'] += 1
            logger.debug(f"清理过期轨迹: ID {gid}")

        expired_cache_keys = [
            key for key, ts in self.invalid_switch_cache.items()
            if current_time - ts > self.trajectory_timeout
        ]
        for key in expired_cache_keys:
            del self.invalid_switch_cache[key]

    def _purge_id_state(self, global_id: int):
        self.trajectory_records.pop(global_id, None)
        self.last_seen_by_id.pop(global_id, None)
        keys_to_remove = [k for k in self.invalid_switch_cache if k[0] == global_id]
        for k in keys_to_remove:
            del self.invalid_switch_cache[k]

    # ================================================================
    # ================================================================

    def _update_tunnel_map_for_merge(self, keep_id: int, merge_id: int):
        try:
            if hasattr(self.tunnel_map, 'trajectory_manager'):
                trajectory_manager = self.tunnel_map.trajectory_manager

                if merge_id in trajectory_manager.global_trajectories:
                    del trajectory_manager.global_trajectories[merge_id]

                if merge_id in trajectory_manager.global_speeds:
                    del trajectory_manager.global_speeds[merge_id]

                if merge_id in trajectory_manager.global_lanes:
                    del trajectory_manager.global_lanes[merge_id]

                logger.debug(f"从隧道地图移除ID {merge_id}的轨迹渲染")

            if hasattr(self.tunnel_map, 'state_manager'):
                state_manager = self.tunnel_map.state_manager

                if merge_id in state_manager.vehicle_states:
                    del state_manager.vehicle_states[merge_id]

                if merge_id in state_manager.motion_params:
                    del state_manager.motion_params[merge_id]

                logger.debug(f"从状态管理器移除ID {merge_id}")

        except Exception as e:
            logger.error(f"更新隧道地图失败: {e}")

    # ================================================================
    # ================================================================

    def _is_valid_camera_transition(self, from_camera: str, to_camera: str) -> bool:
        if from_camera == to_camera:
            return True

        if hasattr(config, 'CAMERA_TOPOLOGY'):
            distance = self._get_topology_distance(from_camera, to_camera)
            return distance is not None

        skip_count = self._get_camera_skip_count(from_camera, to_camera)
        return skip_count <= self.max_camera_skip

    def _get_transition_time_window(self, from_camera: str,
                                    to_camera: str) -> Tuple[float, float]:
        if hasattr(config, 'CAMERA_TRANSITION_TIME_WINDOWS'):
            time_windows = config.CAMERA_TRANSITION_TIME_WINDOWS
            if isinstance(time_windows, dict):
                if (from_camera, to_camera) in time_windows:
                    return time_windows[(from_camera, to_camera)]
                if f"{from_camera}->{to_camera}" in time_windows:
                    return time_windows[f"{from_camera}->{to_camera}"]

        skip_count = self._get_camera_skip_count(from_camera, to_camera)
        max_time = self.camera_switch_max_time * max(skip_count + 1, 1)
        return self.camera_switch_min_time, max_time

    def _get_camera_index(self, camera_id: str) -> Optional[int]:
        try:
            if hasattr(config, "camera_manager") and hasattr(config.camera_manager, "camera_config"):
                order = config.camera_manager.camera_config.get_camera_order()
                if camera_id in order:
                    return order.index(camera_id)
        except Exception:
            pass

        match = re.search(r"(\d+)", camera_id)
        if match:
            return int(match.group(1)) - 1
        return None

    def _get_camera_skip_count(self, from_camera: str, to_camera: str) -> int:
        topology_distance = self._get_topology_distance(from_camera, to_camera)
        if topology_distance is not None:
            return max(topology_distance - 1, 0)

        idx1 = self._get_camera_index(from_camera)
        idx2 = self._get_camera_index(to_camera)
        if idx1 is None or idx2 is None:
            return 0
        return max(abs(idx2 - idx1) - 1, 0)

    def _get_topology_distance(self, from_camera: str, to_camera: str) -> Optional[int]:
        if not hasattr(config, 'CAMERA_TOPOLOGY'):
            return None

        topology = config.CAMERA_TOPOLOGY
        if from_camera not in topology:
            return None

        max_depth = max(self.max_camera_skip + 1, 1)
        visited = {from_camera}
        queue = deque([(from_camera, 0)])

        while queue:
            current, depth = queue.popleft()
            if current == to_camera:
                return depth
            if depth >= max_depth:
                continue
            for neighbor in topology.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return None

    def _estimate_direction(self, from_pos: Tuple[int, int],
                            to_pos: Tuple[int, int]) -> Optional[str]:
        if not from_pos or not to_pos:
            return None
        dx = to_pos[0] - from_pos[0]
        if abs(dx) < 10:
            return "static"
        return "right" if dx > 0 else "left"

    def _estimate_cross_camera_speed_kmh(self, from_pos: Tuple[int, int],
                                         to_pos: Tuple[int, int],
                                         time_gap: float) -> Optional[float]:
        if time_gap <= 0:
            return None

        calibration = getattr(config, "COORDINATE_CALIBRATION", None)
        if not isinstance(calibration, dict):
            return None

        meters_per_pixel_x = float(calibration.get("meters_per_pixel_x", 0.0) or 0.0)
        meters_per_pixel_y = float(calibration.get("meters_per_pixel_y", 0.0) or 0.0)
        if meters_per_pixel_x <= 0 or meters_per_pixel_y <= 0:
            return None

        dx_m = (to_pos[0] - from_pos[0]) * meters_per_pixel_x
        dy_m = (to_pos[1] - from_pos[1]) * meters_per_pixel_y
        distance_m = float(np.sqrt(dx_m * dx_m + dy_m * dy_m))
        speed_m_s = distance_m / time_gap
        return speed_m_s * 3.6
