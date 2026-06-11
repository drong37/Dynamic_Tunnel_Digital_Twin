
import cv2
import numpy as np
import time
import os
import threading
import queue
from datetime import datetime
import argparse
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import config
from vehicle_detector import VehicleDetector
from vehicle_tracker import VehicleTracker
from lane_classifier import LaneClassifier
from camera_calibration import CameraCalibration
from tunnel_map import TunnelMap
from vehicle_reid import VehicleReID
from vehicle_id_corrector import VehicleIDCorrector
from vehicle_prediction_manager import VehiclePredictionManager, VehicleState, MovementDirection


@dataclass
class FrameData:
    camera_id: str
    frame: np.ndarray
    timestamp: float
    original_timestamp: float


@dataclass
class ProcessedResult:
    camera_id: str
    processed_frame: Optional[np.ndarray]
    tracked_vehicles: List[Dict]
    timestamp: float


@dataclass
class SystemStatistics:
    processed_vehicles: int = 0
    filtered_vehicles: int = 0
    blind_zone_predictions: int = 0
    reid_reidentifications: int = 0
    total_frames: int = 0
    start_time: Optional[float] = None


@dataclass
class VehicleBufferEntry:
    patches: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    processed: bool = False


class RTSPBufferedCapture:

    def __init__(self, url: str, buffer_size: int = 8):
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = True
        self.last_frame_time = time.time()
        self.is_connected = False
        self.reconnect_delay = 2.0

        self.consecutive_failures = 0
        self.max_consecutive_failures = 30
        self.reconnect_in_progress = False

        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

    def _reconnect(self):
        if self.reconnect_in_progress:
            time.sleep(0.1)
            return

        self.reconnect_in_progress = True
        print(f"RTSP 主动重连: {self.url}")

        try:
            self.cap.release()
            time.sleep(self.reconnect_delay)
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            self.consecutive_failures = 0
        finally:
            self.reconnect_in_progress = False

    def _reader_thread(self):
        while self.running:
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._reconnect()
                continue

            if not self.cap.isOpened():
                self.is_connected = False
                self._reconnect()
                continue

            ret, frame = self.cap.read()
            if ret:
                self.is_connected = True
                self.last_frame_time = time.time()
                self.consecutive_failures = 0

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                self.consecutive_failures += 1
                self.is_connected = False
                time.sleep(0.01)

    def read(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self.frame_queue.get(timeout=timeout)
            
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            return True, frame
        except queue.Empty:
            return False, None

    def get_time_since_last_frame(self) -> float:
        return time.time() - self.last_frame_time

    def get_consecutive_failures(self) -> int:
        return self.consecutive_failures

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class FrameSynchronizer:

    def __init__(self, camera_ids: List[str], buffer_size: int = 5):
        self.camera_ids = set(camera_ids)
        self.lock = threading.RLock()
        self.new_frame_event = threading.Event()
        self.frames_processed_event = threading.Event()
        self.time_window = 1.0 / 25
        
        self.latest_frames: Dict[str, FrameData] = {}
        self.last_process_time = 0.0
        
        self.processed_buffers: Dict[float, Dict[str, ProcessedResult]] = {}
        self.completed_time_points: set = set()
        self.completed_cameras: set = set()
        
        self.finished_cameras: set = set()
        self.stream_cameras: set = set()

        print(f"帧同步器已初始化: {len(camera_ids)}个摄像头 (非阻塞实时模式)")
    
    def add_frame(self, camera_id: str, frame: Optional[np.ndarray], timestamp: float) -> bool:
        if frame is None:
            return False
            
        with self.lock:
            self.latest_frames[camera_id] = FrameData(
                camera_id, frame, timestamp, timestamp
            )
            self.new_frame_event.set()
            return True
    
    def get_frames_to_process(self) -> Tuple[Optional[float], Optional[Dict]]:
        with self.lock:
            current_time = time.time()
            
            if current_time - self.last_process_time < self.time_window:
                return None, None
            
            if not self.latest_frames:
                return None, None
                
            if len(self.latest_frames) < max(1, len(self.camera_ids) // 2):
                return None, None
                
            self.last_process_time = current_time
            return current_time, self.latest_frames.copy()
    
    def add_processed_result(self, time_point: float, camera_id: str,
                           processed_frame: Optional[np.ndarray], tracked_vehicles: List) -> bool:
        with self.lock:
            if time_point not in self.processed_buffers:
                self.processed_buffers[time_point] = {}

            self.processed_buffers[time_point][camera_id] = ProcessedResult(
                camera_id, processed_frame, tracked_vehicles, time_point
            )

            if len(self.processed_buffers[time_point]) >= len(self.latest_frames):
                self.completed_time_points.add(time_point)
                self.frames_processed_event.set()
                return True
        return False
    
    def get_latest_processed_results(self) -> Tuple[Optional[float], Optional[Dict]]:
        with self.lock:
            completed_points = sorted(list(self.completed_time_points))
            
            if not completed_points:
                return None, None
            
            latest_point = completed_points[-1]
            
            for tp in completed_points[:-1]:
                self.processed_buffers.pop(tp, None)
                self.completed_time_points.remove(tp)
            
            return latest_point, self.processed_buffers.get(latest_point, {}).copy()
            
    def mark_stream_camera(self, camera_id: str):
        with self.lock:
            self.stream_cameras.add(camera_id)

    def notify_camera_finished(self, camera_id: str):
        with self.lock:
            self.finished_cameras.add(camera_id)
            self.latest_frames.pop(camera_id, None)
            self.new_frame_event.set()
            self.frames_processed_event.set()

    def is_all_completed(self) -> bool:
        with self.lock:
            file_cameras = self.camera_ids - self.stream_cameras
            if not file_cameras:
                return False
            return file_cameras.issubset(self.finished_cameras)

    def tick(self) -> bool:
        self.new_frame_event.set()
        return True

    def notify_no_frame(self, camera_id: str, current_time: float):
        pass
        
    def _cleanup_old_buffers(self):
        pass

    def _try_trigger_processing(self, current_time: float):
        pass


class VideoCapture:
    
    def __init__(self, camera_config: Dict):
        self.camera_config = camera_config
        self.captures: Dict[str, Any] = {}
        self.video_fps: Dict[str, float] = {}
    
    def open_videos(self) -> bool:
        success = True
        for cam_id, cam_config in self.camera_config.items():
            video_path = cam_config['video_path']
            
            if video_path.startswith(('rtsp://', 'http://', 'https://', 'rtmp://')):
                print(f"检测到实时流 {video_path}，将在子线程中延迟连接")
                self.captures[cam_id] = None
                self.video_fps[cam_id] = config.TARGET_FPS
                continue
            
            print(f"打开视频: {video_path}")
            
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                print(f"错误: 无法打开视频 {video_path}")
                success = False
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self.video_fps[cam_id] = fps
            self.captures[cam_id] = cap
            
            print(f"视频 {cam_id}: {fps} FPS, {total_frames} 帧, {duration:.2f} 秒")
        
        return success
    
    def close_videos(self):
        for cap in self.captures.values():
            cap.release()
        self.captures.clear()


class EnhancedVehicleIDManager:
    
    def __init__(self, reid_system: VehicleReID, tunnel_map: TunnelMap):
        self.reid_system = reid_system
        self.tunnel_map = tunnel_map
        self.vehicle_id_mapping: Dict[Tuple[str, int], int] = {}  # (camera_id, local_id) -> global_id
        self.vehicle_last_seen: Dict[Tuple[str, int], float] = {}
        self.id_mapping_ttl = 120.0
        
        self.vehicle_buffer: Dict[Tuple[str, int], VehicleBufferEntry] = {}
        self.buffer_size = 3
        self.buffer_timeout = 2.0
        self.quality_threshold = 0.3
        
        self.id_corrector = VehicleIDCorrector(tunnel_map)
        self.id_corrector.set_correction_callback(self._handle_correction_event)

        self.blocked_global_ids: Dict[Tuple[str, int], float] = {}
        self.block_duration = 6.0
        
        self.last_independent_correction_check = 0.0
        self.independent_correction_interval = 10.0

        active_cfg = config.TRACKING_STRATEGY['active_tracking']
        self.active_mapping_window = float(active_cfg['mapping_recent_sec'])
        self.active_detection_recent_window = float(active_cfg['detection_recent_sec'])
        self.active_detection_stale_window = float(active_cfg['detection_stale_sec'])
        self.camera_active_window = float(active_cfg['camera_recent_sec'])
        
        self.prediction_manager = VehiclePredictionManager(tunnel_map)
        
        print("增强版车辆ID管理器已初始化，支持ID纠错和缓冲机制，使用统一的预测管理器")
    
    def update_vehicle_ids(self, camera_id: str, tracked_vehicles: List, 
                        time_point: float) -> Tuple[Dict[int, int], List[Tuple]]:
        detection_region = config.DETECTION_REGIONS.get(camera_id, [])
        global_ids = {}
        map_updates = []
        
        camera_order = config.camera_manager.camera_config.get_camera_order()
        try:
            cam_index = camera_order.index(camera_id)
            cam_prefix = (cam_index + 1) * 10000  # cam1=10000, cam2=20000, ..., cam6=60000
        except ValueError:
            cam_prefix = 0
        
        active_local_ids = set()
        detected_global_ids = set()
        
        for vehicle in tracked_vehicles:
            local_id = vehicle['local_track_id']
            bbox = vehicle.get('bbox', [0, 0, 0, 0])
            vehicle_patch = vehicle.get('patch', None)
            in_detection_region = vehicle.get('in_detection_region')
            class_id = vehicle.get('class_id', None)

            track_key = (camera_id, local_id)
            active_local_ids.add(track_key)
            fallback_id = cam_prefix + local_id

            global_id = self._process_reid(
                track_key, vehicle_patch, in_detection_region,
                camera_id, time_point, fallback_id, bbox
            )

            if global_id is None:
                if track_key not in self.vehicle_id_mapping:
                    continue
                global_id = self.vehicle_id_mapping[track_key]
                self.vehicle_last_seen[track_key] = time_point

            corrector_pos = self.tunnel_map.bbox_to_map_position(camera_id, bbox)
            if corrector_pos is not None:
                accepted = self.id_corrector.update_vehicle_trajectory(
                    global_id, corrector_pos, time_point, camera_id, float(corrector_pos[1]),
                    local_id=local_id
                )
                if not accepted:
                    continue

            self.prediction_manager.update_vehicle_tracking_info(global_id, camera_id, bbox, time_point)
            detected_global_ids.add(global_id)
            global_ids[local_id] = global_id
            map_updates.append((global_id, camera_id, bbox, time_point, class_id))
        
        self.prediction_manager.handle_lost_vehicles(time_point, detected_global_ids)
        
        self._execute_independent_correction_check(time_point)
        
        self._cleanup_vehicle_mappings(time_point, active_local_ids, camera_id)
        self.reid_system.cleanup_gallery(max_age=int(self.id_mapping_ttl))
        
        return global_ids, map_updates
    

    
    def _cleanup_vehicle_mappings(self, current_time: float, active_ids: set, camera_id: str):
        keys_to_remove = []
        
        for key, last_seen in self.vehicle_last_seen.items():
            if camera_id is not None and key[0] != camera_id:
                continue
            if key in active_ids:
                continue
            if current_time - last_seen > self.id_mapping_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.vehicle_id_mapping.pop(key, None)
            self.vehicle_last_seen.pop(key, None)
    
    def _add_to_buffer(self, track_key: Tuple[str, int], vehicle_patch: np.ndarray, 
                      bbox: List[float], timestamp: float) -> bool:
        if vehicle_patch is None:
            return False
        
        quality_score = self.reid_system.quality_assessor.assess_image_quality(vehicle_patch)
        
        if quality_score < self.quality_threshold:
            return False
        
        if track_key not in self.vehicle_buffer:
            self.vehicle_buffer[track_key] = VehicleBufferEntry(
                first_seen=timestamp,
                last_seen=timestamp
            )
        
        buffer_entry = self.vehicle_buffer[track_key]
        
        if buffer_entry.processed:
            return False
        
        buffer_entry.patches.append(vehicle_patch.copy())
        buffer_entry.timestamps.append(timestamp)
        buffer_entry.quality_scores.append(quality_score)
        buffer_entry.bboxes.append(bbox.copy())
        buffer_entry.last_seen = timestamp
        
        if len(buffer_entry.patches) > self.buffer_size:
            buffer_entry.patches.pop(0)
            buffer_entry.timestamps.pop(0)
            buffer_entry.quality_scores.pop(0)
            buffer_entry.bboxes.pop(0)
        
        return True
    
    def _should_process_buffer(self, track_key: Tuple[str, int], in_detection_region: bool, 
                             current_time: float) -> bool:
        if track_key not in self.vehicle_buffer:
            return False
        
        buffer_entry = self.vehicle_buffer[track_key]
        
        if buffer_entry.processed:
            return False
        
        if len(buffer_entry.patches) == 0:
            return False
        
        if len(buffer_entry.patches) >= self.buffer_size:
            return True
        
        if not in_detection_region and len(buffer_entry.patches) > 0:
            return True
        
        if current_time - buffer_entry.first_seen > self.buffer_timeout:
            return True
        
        return False
    
    def _process_buffer(self, track_key: Tuple[str, int], camera_id: str, 
                       current_time: float) -> Optional[int]:
        if track_key not in self.vehicle_buffer:
            return None
        
        buffer_entry = self.vehicle_buffer[track_key]
        
        if buffer_entry.processed or len(buffer_entry.patches) == 0:
            return None
        
        try:
            buffer_entry.processed = True
            
            best_patch = self._extract_best_quality_patch(buffer_entry)
            
            if best_patch is None:
                print(f"无法从缓冲区提取有效图像块: {track_key}")
                return None
            
            if track_key in self.vehicle_id_mapping:
                existing_id = self.vehicle_id_mapping[track_key]
                self.vehicle_last_seen[track_key] = current_time

                if self._is_global_id_blocked(camera_id, existing_id, current_time):
                    self._invalidate_track_mapping(track_key)
                    existing_id = None
                
                if existing_id is not None and not self.reid_system.update_gallery(
                    existing_id, best_patch, camera_id=camera_id
                ):
                    reid_global_id = self.reid_system.get_global_id(best_patch, camera_id)
                    if reid_global_id is not None:
                        if self._is_global_id_blocked(camera_id, reid_global_id, current_time):
                            forced_id = self._force_new_global_id(best_patch, camera_id)
                            if forced_id is None:
                                print(f"缓冲区ReID被阻断且无法新建ID: {track_key}")
                                return None
                            reid_global_id = forced_id
                        self.vehicle_id_mapping[track_key] = reid_global_id
                        self.vehicle_last_seen[track_key] = current_time
                        print(f"缓冲区ReID成功: {track_key} -> 新ID {reid_global_id}")
                        return reid_global_id
                    else:
                        print(f"缓冲区ReID失败: {track_key}")
                        return None
                elif existing_id is not None:
                    print(f"缓冲区特征更新成功: {track_key} -> ID {existing_id}")
                    return existing_id
                else:
                    reid_global_id = self.reid_system.get_global_id(best_patch, camera_id)
                    if reid_global_id is not None:
                        if self._is_global_id_blocked(camera_id, reid_global_id, current_time):
                            forced_id = self._force_new_global_id(best_patch, camera_id)
                            if forced_id is None:
                                print(f"缓冲区ReID被阻断且无法新建ID: {track_key}")
                                return None
                            reid_global_id = forced_id
                        self.vehicle_id_mapping[track_key] = reid_global_id
                        self.vehicle_last_seen[track_key] = current_time
                        print(f"缓冲区ReID成功: {track_key} -> 新ID {reid_global_id}")
                        return reid_global_id
                    print(f"缓冲区ReID失败: {track_key}")
                    return None
            else:
                reid_global_id = self.reid_system.get_global_id(best_patch, camera_id)
                if reid_global_id is not None:
                    if self._is_global_id_blocked(camera_id, reid_global_id, current_time):
                        forced_id = self._force_new_global_id(best_patch, camera_id)
                        if forced_id is None:
                            print(f"缓冲区ReID被阻断且无法新建ID: {track_key}")
                            return None
                        reid_global_id = forced_id
                    self.vehicle_id_mapping[track_key] = reid_global_id
                    self.vehicle_last_seen[track_key] = current_time
                    print(f"缓冲区ReID成功: {track_key} -> 新ID {reid_global_id}")
                    return reid_global_id
                else:
                    print(f"缓冲区ReID失败: {track_key}")
                    return None
        
        except Exception as e:
            print(f"处理缓冲区时出错: {e}")
            return None
    
    def _extract_best_quality_patch(self, buffer_entry: VehicleBufferEntry) -> Optional[np.ndarray]:
        if len(buffer_entry.patches) == 0:
            return None
        
        best_index = np.argmax(buffer_entry.quality_scores)
        best_patch = buffer_entry.patches[best_index]
        best_quality = buffer_entry.quality_scores[best_index]
        
        print(f"选择最佳图像块: 质量分数={best_quality:.3f}, 总数={len(buffer_entry.patches)}")
        
        return best_patch
    
    def _extract_average_features(self, buffer_entry: VehicleBufferEntry) -> Optional[np.ndarray]:
        if len(buffer_entry.patches) == 0:
            return None
        
        features = []
        weights = []
        
        for i, patch in enumerate(buffer_entry.patches):
            feature = self.reid_system.extract_features(patch)
            if feature is not None:
                features.append(feature)
                weights.append(buffer_entry.quality_scores[i])
        
        if len(features) == 0:
            return None
        
        features_array = np.array(features)
        weights_array = np.array(weights)
        
        weights_array = weights_array / np.sum(weights_array)
        
        average_features = np.average(features_array, axis=0, weights=weights_array)
        
        norm = np.linalg.norm(average_features)
        if norm > 0:
            average_features = average_features / norm
        
        return average_features
    
    def _cleanup_expired_buffers(self, current_time: float):
        keys_to_remove = []
        
        for track_key, buffer_entry in self.vehicle_buffer.items():
            if current_time - buffer_entry.last_seen > self.buffer_timeout * 2:
                keys_to_remove.append(track_key)
        
        for key in keys_to_remove:
            del self.vehicle_buffer[key]
            print(f"清理过期缓冲区: {key}")
    
    def _process_reid(self, track_key: Tuple[str, int], vehicle_patch: Optional[np.ndarray], 
                      in_detection_region: bool, camera_id: str, time_point: float, 
                      fallback_id: int, bbox: List[float]) -> Optional[int]:
        
        self._cleanup_expired_buffers(time_point)
        self._cleanup_expired_blocks(time_point)
        
        existing_global_id = self.vehicle_id_mapping.get(track_key)
        if existing_global_id is not None and self._is_global_id_blocked(camera_id, existing_global_id, time_point):
            self._invalidate_track_mapping(track_key)
            existing_global_id = None
        
        if in_detection_region and vehicle_patch is not None:
            self._add_to_buffer(track_key, vehicle_patch, bbox, time_point)
            
            if self._should_process_buffer(track_key, in_detection_region, time_point):
                reid_result = self._process_buffer(track_key, camera_id, time_point)
                if reid_result is not None:
                    return reid_result
                else:
                    if existing_global_id is not None:
                        self.vehicle_last_seen[track_key] = time_point
                        return existing_global_id
                    else:
                        return None
            else:
                if existing_global_id is not None:
                    self.vehicle_last_seen[track_key] = time_point
                    return existing_global_id
                else:
                    return None
        
        elif not in_detection_region and existing_global_id is not None:
            if self._should_process_buffer(track_key, in_detection_region, time_point):
                reid_result = self._process_buffer(track_key, camera_id, time_point)
                if reid_result is not None:
                    return reid_result
            
            self.vehicle_last_seen[track_key] = time_point
            return existing_global_id
        
        return None

    def _handle_correction_event(self, event):
        try:
            action = getattr(event, "action", None)
            if action != "split_id":
                return

            global_id = getattr(event, "global_id", None)
            camera_id = getattr(event, "camera_id", None)
            track_key = getattr(event, "track_key", None)

            if track_key is not None:
                self._invalidate_track_mapping(track_key)

            if camera_id is not None and global_id is not None:
                self._block_global_id_for_camera(camera_id, global_id, self.block_duration)
        except Exception as e:
            print(f"纠错事件处理失败: {e}")

    def _invalidate_track_mapping(self, track_key: Tuple[str, int]):
        if track_key in self.vehicle_id_mapping:
            del self.vehicle_id_mapping[track_key]
        self.vehicle_last_seen.pop(track_key, None)
        if track_key in self.vehicle_buffer:
            del self.vehicle_buffer[track_key]

    def _block_global_id_for_camera(self, camera_id: str, global_id: int, duration: float):
        expire_time = time.time() + duration
        self.blocked_global_ids[(camera_id, global_id)] = expire_time

    def _is_global_id_blocked(self, camera_id: str, global_id: int, current_time: float) -> bool:
        expire_time = self.blocked_global_ids.get((camera_id, global_id))
        if expire_time is None:
            return False
        return current_time <= expire_time

    def _cleanup_expired_blocks(self, current_time: float):
        expired = [key for key, expire in self.blocked_global_ids.items() if expire <= current_time]
        for key in expired:
            del self.blocked_global_ids[key]

    def _force_new_global_id(self, vehicle_patch: np.ndarray, camera_id: str) -> Optional[int]:
        try:
            features = self.reid_system.extract_features(vehicle_patch)
            if features is None:
                return None
            quality_score = self.reid_system.quality_assessor.assess_image_quality(vehicle_patch)
            new_id = self.reid_system._create_new_vehicle_entry(features, quality_score, camera_id)
            return new_id
        except Exception as e:
            print(f"强制创建新ID失败: {e}")
            return None
    
    def _execute_independent_correction_check(self, current_time: float):
        if current_time - self.last_independent_correction_check >= self.independent_correction_interval:
            try:
                self.id_corrector.execute_correction_cycle(current_time)
                self.last_independent_correction_check = current_time
                
                if int(current_time) % 30 == 0:
                    stats = self.id_corrector.get_correction_statistics()
                    if stats['total_corrections'] > 0:
                        print(f"[纠错统计] 总纠正: {stats['total_corrections']}, " +
                              f"成功合并: {stats['successful_merges']}, " +
                              f"活跃轨迹: {stats['active_trajectories']}")
                        
            except Exception as e:
                print(f"独立纠错检查失败: {e}")
    
    def is_global_id_actively_tracked(self, global_id: int, camera_id: str,
                                    current_time: Optional[float] = None) -> bool:
        try:
            now = current_time if current_time is not None else time.time()

            found_in_mapping = False
            for track_key, mapped_id in self.vehicle_id_mapping.items():
                if mapped_id == global_id:
                    if track_key in self.vehicle_last_seen:
                        time_since_seen = now - self.vehicle_last_seen[track_key]
                        if time_since_seen < self.active_mapping_window:
                            found_in_mapping = True
                            break
            
            if global_id in self.prediction_manager.vehicle_tracking_info:
                info = self.prediction_manager.vehicle_tracking_info[global_id]
                time_since_detection = now - info.last_detection_time
                
                if time_since_detection < self.active_detection_recent_window and found_in_mapping:
                    return True
                    
                if time_since_detection > self.active_detection_stale_window:
                    return False
            
            return found_in_mapping and global_id in self.prediction_manager.vehicle_tracking_info
            
        except Exception as e:
            print(f"检查全局ID活跃状态失败 {global_id}: {e}")
            return False
    
    def get_actively_tracked_ids_in_camera(self, camera_id: str,
                                        current_time: Optional[float] = None) -> set:
        active_ids = set()
        try:
            now = current_time if current_time is not None else time.time()
            
            for track_key, global_id in self.vehicle_id_mapping.items():
                key_camera_id, local_id = track_key
                
                if key_camera_id == camera_id:
                    if track_key in self.vehicle_last_seen:
                        time_since_seen = now - self.vehicle_last_seen[track_key]
                        
                        if time_since_seen < self.camera_active_window:
                            active_ids.add(global_id)
            
            return active_ids
            
        except Exception as e:
            print(f"获取摄像头 {camera_id} 活跃ID失败: {e}")
            return set()
    
    def cleanup_ghost_vehicle_mapping(self, ghost_id: int):
        try:
            keys_to_remove = [
                track_key for track_key, mapped_id in list(self.vehicle_id_mapping.items())
                if mapped_id == ghost_id
            ]
            buffer_keys_to_remove = set(keys_to_remove)

            for key in keys_to_remove:
                del self.vehicle_id_mapping[key]
                self.vehicle_last_seen.pop(key, None)
                print(f"清理幽灵车辆 {ghost_id} 的ID映射: {key}")
            
            if ghost_id in self.prediction_manager.vehicle_tracking_info:
                del self.prediction_manager.vehicle_tracking_info[ghost_id]
                print(f"清理幽灵车辆 {ghost_id} 的跟踪信息")
            
            for key in buffer_keys_to_remove:
                if key in self.vehicle_buffer:
                    del self.vehicle_buffer[key]
                    print(f"清理幽灵车辆 {ghost_id} 的缓冲区: {key}")
                
        except Exception as e:
            print(f"清理幽灵车辆映射失败 {ghost_id}: {e}")


class VideoOutputManager:
    
    def __init__(self, output_width: int, output_height: int):
        self.output_width = output_width
        self.output_height = output_height
        self.camera_writers: Dict[str, cv2.VideoWriter] = {}
        self.integrated_writer: Optional[cv2.VideoWriter] = None
        self.camera_output_paths: Dict[str, str] = {}
        self.integrated_output_path: Optional[str] = None
    
    def setup_output_paths(self):
        output_dir = os.path.join(config.OUTPUT_DIR, 'processed_videos')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.integrated_output_path = os.path.join(
            output_dir, f"tunnel_monitoring_{timestamp}_{self.output_width}x{self.output_height}.mp4"
        )
        
        for cam_id in config.CAMERA_CONFIG:
            self.camera_output_paths[cam_id] = os.path.join(
                output_dir, f"camera_{cam_id}_{timestamp}.mp4"
            )
    
    def write_camera_video(self, cam_id: str, frame: np.ndarray):
        if cam_id not in self.camera_output_paths or frame is None:
            return
        
        if cam_id not in self.camera_writers or self.camera_writers[cam_id] is None:
            self._initialize_camera_writer(cam_id, frame)
        
        if self.camera_writers[cam_id]:
            self.camera_writers[cam_id].write(frame)
    
    def write_integrated_video(self, frame: np.ndarray):
        if frame is None or not self.integrated_output_path:
            return
        
        if self.integrated_writer is None:
            self._initialize_integrated_writer()
        
        if self.integrated_writer:
            self.integrated_writer.write(frame)
    
    def _initialize_camera_writer(self, cam_id: str, frame: np.ndarray):
        output_path = self.camera_output_paths[cam_id]
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (w, h), True)
        
        if not writer.isOpened():
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, (w, h))
        
        self.camera_writers[cam_id] = writer
    
    def _initialize_integrated_writer(self):
        if not self.integrated_output_path:
            return
            
        fourcc = cv2.VideoWriter.fourcc(*'avc1')
        writer = cv2.VideoWriter(
            self.integrated_output_path, fourcc, config.TARGET_FPS,
            (self.output_width, self.output_height), True
        )
        
        if not writer.isOpened():
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                self.integrated_output_path, fourcc, config.TARGET_FPS,
                (self.output_width, self.output_height)
            )
        
        self.integrated_writer = writer
    
    def close_all_writers(self):
        for cam_id, writer in self.camera_writers.items():
            if writer and writer.isOpened():
                writer.release()
                output_path = self.camera_output_paths.get(cam_id, "unknown")
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"摄像头 {cam_id} 视频已保存: {output_path} ({file_size:.2f} MB)")
        
        if self.integrated_writer and self.integrated_writer.isOpened():
            self.integrated_writer.release()
            print(f"整合视图视频已保存: {self.integrated_output_path}")


class ViewComposer:
    
    def __init__(self, output_width: int, output_height: int):
        self.output_width = output_width
        self.output_height = output_height
    
    def create_combined_view(self, camera_frames: Dict[str, np.ndarray], 
                           map_frame: np.ndarray, timestamp: float) -> np.ndarray:
        map_h, map_w = map_frame.shape[:2]
        
        sorted_cam_ids = sorted(camera_frames.keys())
        valid_frames = [(cam_id, frame) for cam_id, frame in 
                       [(c, camera_frames.get(c)) for c in sorted_cam_ids] if frame is not None]
        
        if not valid_frames:
            return self._resize_to_output(map_frame)
        
        camera_panel = self._create_camera_panel(valid_frames, map_w, timestamp)
        
        combined_view = cv2.vconcat([camera_panel, map_frame])
        return self._resize_to_output(combined_view)
    
    def _create_camera_panel(self, valid_frames: List, map_w: int, timestamp: float) -> np.ndarray:
        num_cameras = len(valid_frames)
        cols = 3
        rows = (num_cameras + cols - 1) // cols
        camera_width = map_w // cols
        
        _, first_frame = valid_frames[0]
        first_h, first_w = first_frame.shape[:2]
        aspect_ratio = first_h / first_w
        camera_height = int(camera_width * aspect_ratio)
        
        total_height = camera_height * rows
        camera_panel = np.zeros((total_height, map_w, 3), dtype=np.uint8)
        
        for i, (cam_id, frame) in enumerate(valid_frames):
            row = i // cols
            col = i % cols
            x_pos = col * camera_width
            y_pos = row * camera_height
            
            resized = cv2.resize(frame, (camera_width, camera_height))
            camera_panel[y_pos:y_pos+camera_height, x_pos:x_pos+camera_width] = resized
            
            self._add_camera_label(camera_panel, cam_id, timestamp, x_pos, y_pos)
        
        return camera_panel

    def _add_camera_label(self, camera_panel: np.ndarray, cam_id: str, 
                        timestamp: float, x_pos: int, y_pos: int = 0):
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        label = f"{cam_id} - ({timestamp_str})"
        
        overlay = camera_panel.copy()
        cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + 300, y_pos + 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, camera_panel, 0.5, 0, camera_panel)
        
        cv2.putText(camera_panel, label, (x_pos + 10, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    def _resize_to_output(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]
        final_output = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        scale_factor = min(self.output_width / original_w, self.output_height / original_h)
        resized_width = int(original_w * scale_factor)
        resized_height = int(original_h * scale_factor)
        
        resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
        
        start_x = (self.output_width - resized_width) // 2
        start_y = (self.output_height - resized_height) // 2
        
        final_output[start_y:start_y+resized_height, start_x:start_x+resized_width] = resized_image
        
        return final_output


class EnhancedVehicleSystem:
    
    def __init__(self, output_width: int = 1920, output_height: int = 1080):
        self.output_width = output_width
        self.output_height = output_height
        self.is_running = False
        self.speed_factor = 1.0
        
        self.statistics = SystemStatistics()
        
        self._initialize_components()
        
        self.capture_threads: Dict[str, threading.Thread] = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        self.ws_push_thread: Optional[threading.Thread] = None
        self.incremental_push_thread: Optional[threading.Thread] = None
        self.ws_push_flag = threading.Event()
        self.processing_times: Dict[str, List[float]] = defaultdict(list)
        self.processing_fps: Dict[str, float] = {}
        
        print("增强版系统初始化完成! 支持丢失车辆预测和图像缓冲机制")
    
    def _initialize_components(self):
        print("初始化系统组件...")
        
        self.detector = VehicleDetector(
            model_path=config.YOLO_MODEL_PATH,
            conf_threshold=config.CONFIDENCE_THRESHOLD,
            vehicle_classes=config.VEHICLE_CLASSES
        )
        
        self.trackers = {}
        self.lane_classifiers = {}
        self.calibrations = {}
        
        for cam_id, cam_config in config.CAMERA_CONFIG.items():
            self.trackers[cam_id] = VehicleTracker(
                camera_id=cam_id,
                strongsort_config_path=config.STRONGSORT_CONFIG_PATH,
                strongsort_weights=config.STRONGSORT_WEIGHTS
            )
            self.lane_classifiers[cam_id] = LaneClassifier(cam_id)
            self.calibrations[cam_id] = CameraCalibration(cam_id)
        
        self.tunnel_map = TunnelMap(
            map_width=config.MAP_WIDTH,
            map_height=config.MAP_HEIGHT,
            camera_calibrations=self.calibrations
        )
        
        model_path = os.path.join(config.MODEL_DIR, "deit_base_distilled_patch16_224-df68dfff.pth")
        test_path = os.path.join(config.MODEL_DIR, "deit_C2T-ReID_vehicleID.pth")
        config_path = os.path.join(config.BASE_DIR, "C2T-ReID/configs/VehicleID/deit_C2T-ReID_stride.yml")
        
        self.reid_system = VehicleReID(
            model_path,
            config_path,
            test_path,
            matching_threshold=config.REID_MATCH_THRESHOLD,
            camera_topology=config.CAMERA_TOPOLOGY
        )
        
        self.id_manager = EnhancedVehicleIDManager(self.reid_system, self.tunnel_map)
        
        self.tunnel_map.set_id_manager(self.id_manager)
        
        self.tunnel_map.set_prediction_manager(self.id_manager.prediction_manager)
        
        self.video_capture = VideoCapture(config.CAMERA_CONFIG)
        self.output_manager = VideoOutputManager(self.output_width, self.output_height)
        self.view_composer = ViewComposer(self.output_width, self.output_height)

        self.frame_synchronizer: Optional[FrameSynchronizer] = None
    
    def start(self, output_video: bool = True):
        if self.is_running:
            print("系统已在运行中")
            return
        
        self.is_running = True
        self.statistics.start_time = time.time()
        
        if output_video:
            self.output_manager.setup_output_paths()
        
        if not self.video_capture.open_videos():
            print("视频打开失败")
            return
        
        self.frame_synchronizer = FrameSynchronizer(list(self.video_capture.captures.keys()))
        
        self._start_threads()

        if self.ws_push_manager:
            self.ws_push_manager.start()

        print("增强版系统已启动 (含图像缓冲机制). 按ESC退出, '+'/'-'调整播放速度")
    
    def _start_threads(self):
        print("正在启动摄像头线程 (后台交错启动)...")
        
        sorted_cameras = sorted(self.video_capture.captures.items())
        
        for index, (cam_id, cap) in enumerate(sorted_cameras):
            start_delay = index * 1.5
            
            thread = threading.Thread(
                target=self._capture_thread,
                args=(cam_id, cap, start_delay),
                daemon=True
            )
            self.capture_threads[cam_id] = thread
            thread.start()
            self.processing_fps[cam_id] = 0

        self.processing_thread = threading.Thread(target=self._process_thread, daemon=True)

        self.processing_thread.start()

        if self.ws_push_manager:
            self.incremental_push_thread = threading.Thread(
                target=self._incremental_push_thread,
                name="IncrementalPush",
                daemon=True
            )
            self.incremental_push_thread.start()

            # self.ws_push_thread = threading.Thread(
            #     target=self._ws_push_thread,
            #     name="WebSocketPush",
            #     daemon=True
            # )
            # self.ws_push_thread.start()
    
    def _capture_thread(self, camera_id: str, capture: cv2.VideoCapture, start_delay: float = 0.0):
        
        video_path = config.CAMERA_CONFIG[camera_id]['video_path']
        is_rtsp_stream = video_path.startswith(('rtsp://', 'http://', 'https://', 'rtmp://'))

        if start_delay > 0 and is_rtsp_stream:
            print(f"摄像头 {camera_id} 将在 {start_delay:.1f} 秒后连接...")
            time.sleep(start_delay)
            
        assert self.frame_synchronizer is not None
        original_fps = self.video_capture.video_fps.get(camera_id, 30.0)
        target_fps = config.TARGET_FPS
        skip_frames = max(1, int(original_fps / target_fps))

        target_frame_interval = 1.0 / target_fps

        start_time = time.time()
        frame_count = 0
        processed_count = 0
        frame_index = 0
        next_frame_time = start_time

        print(f"摄像头 {camera_id}: 原始 {original_fps} FPS → 目标 {target_fps} FPS")

        rtsp_capture: Optional[RTSPBufferedCapture] = None
        if is_rtsp_stream:
            rtsp_capture = RTSPBufferedCapture(video_path, buffer_size=8)
            self.frame_synchronizer.mark_stream_camera(camera_id)
            print(f"摄像头 {camera_id}: 使用 RTSP 缓冲读取器（buffer_size=8）")

        last_warning_time = 0.0
        frame_log_interval = 100

        try:
            while self.is_running:
                current_time = time.time()

                if not is_rtsp_stream:
                    if current_time < next_frame_time and self.speed_factor == 1.0:
                        time.sleep(0.001)
                        continue

                if is_rtsp_stream:
                    ret, frame = rtsp_capture.read(timeout=0.1)
                    if not ret:
                        self.frame_synchronizer.notify_no_frame(camera_id, time.time())

                        time_since_last = rtsp_capture.get_time_since_last_frame()
                        if time_since_last > 5.0 and current_time - last_warning_time > 5.0:
                            print(f"摄像头 {camera_id}: RTSP 已 {time_since_last:.1f} 秒无帧，可能断连（连续失败: {rtsp_capture.get_consecutive_failures()}）")
                            last_warning_time = current_time
                        continue

                    if ret:
                        if processed_count % frame_log_interval == 0 and processed_count > 0:
                            print(f"摄像头 {camera_id}: 已处理 {processed_count} 帧")
                    else:
                        time_since_last = rtsp_capture.get_time_since_last_frame()
                        if time_since_last > 5.0 and current_time - last_warning_time > 5.0:
                            print(f"摄像头 {camera_id}: RTSP 已 {time_since_last:.1f} 秒无帧，可能断连（连续失败: {rtsp_capture.get_consecutive_failures()}）")
                            last_warning_time = current_time
                        continue
                else:
                    ret, frame = capture.read()
                    if not ret:
                        print(f"视频 {camera_id} 播放完成")
                        self.frame_synchronizer.notify_camera_finished(camera_id)
                        break

                frame_count += 1
                frame_index += 1

                if frame_index % skip_frames == 0:
                    if is_rtsp_stream:
                        video_timestamp = time.time()
                    else:
                        target_time = processed_count * target_frame_interval
                        video_timestamp = start_time + target_time

                    self.frame_synchronizer.add_frame(camera_id, frame, video_timestamp)
                    processed_count += 1

                    next_frame_time = start_time + ((processed_count + 1) * target_frame_interval / self.speed_factor)
                    self.statistics.total_frames += 1
        finally:
            if rtsp_capture:
                rtsp_capture.release()
    
    def _process_thread(self):
        assert self.frame_synchronizer is not None
        while self.is_running:
            waited = self.frame_synchronizer.new_frame_event.wait(timeout=0.1)
            self.frame_synchronizer.new_frame_event.clear()

            if not waited:
                self.frame_synchronizer.tick()

            time_point, frames = self.frame_synchronizer.get_frames_to_process()
            if time_point is None or not frames:
                continue
            
            map_updates = []
            
            for camera_id, frame_data in frames.items():
                if frame_data.frame is None:
                    self.frame_synchronizer.add_processed_result(time_point, camera_id, None, [])
                    continue
                
                processed_frame, tracked_vehicles = self._process_single_camera(
                    camera_id, frame_data.frame, time_point
                )
                
                global_ids, camera_map_updates = self.id_manager.update_vehicle_ids(
                    camera_id, tracked_vehicles, time_point
                )
                map_updates.extend(camera_map_updates)
                
                self.frame_synchronizer.add_processed_result(
                    time_point, camera_id, processed_frame, tracked_vehicles
                )
            
            self._update_tunnel_map(map_updates)
            
            self._update_processing_statistics()
    
    def _process_single_camera(self, camera_id: str, frame: np.ndarray, 
                             timestamp: float) -> Tuple[np.ndarray, List]:
        process_start = time.time()
        
        tracker = self.trackers[camera_id]
        detections = self.detector.detect(frame)
        tracked_vehicles = tracker.update(frame, detections)
        
        result_frame = tracker.draw_tracks(frame, tracked_vehicles, {})
        
        processing_time = time.time() - process_start
        self.processing_times[camera_id].append(processing_time)
        if len(self.processing_times[camera_id]) > 100:
            self.processing_times[camera_id] = self.processing_times[camera_id][-100:]
        
        if self.processing_times[camera_id]:
            avg_time = sum(self.processing_times[camera_id]) / len(self.processing_times[camera_id])
            self.processing_fps[camera_id] = 1.0 / avg_time if avg_time > 0 else 0
        
        return result_frame, tracked_vehicles
    
    def _update_tunnel_map(self, map_updates: List):
        for update_tuple in map_updates:
            try:
                if len(update_tuple) == 5:
                    global_id, camera_id, bbox, update_time, class_id = update_tuple
                else:
                    global_id, camera_id, bbox, update_time = update_tuple
                    class_id = 2

                self.tunnel_map.update_vehicle_position(global_id, camera_id, bbox, update_time, class_id)

            except Exception as e:
                print(f"地图更新错误: {e}")
    
    def _update_processing_statistics(self):
        if self.statistics.total_frames % 100 == 0:
            in_region = self.statistics.processed_vehicles - self.statistics.filtered_vehicles
            print(f"处理帧数: {self.statistics.total_frames}, "
                  f"车辆总数: {self.statistics.processed_vehicles}, "
                  f"区域内: {in_region}")

    def _incremental_push_thread(self):
        while self.is_running:
            try:
                if self.ws_push_manager:
                    self.ws_push_manager.push_incremental()
            except Exception as e:
                print(f"增量推送异常: {e}")
            time.sleep(0.01)

    def _run_display_loop(self):
        assert self.frame_synchronizer is not None
        window_name = "Tunnel Monitoring System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.output_width, self.output_height)

        target_frame_time = 1.0 / config.TARGET_FPS
        last_display_time = time.time()
        last_combined_view = None

        while self.is_running:
            waited = self.frame_synchronizer.frames_processed_event.wait(timeout=0.05)
            self.frame_synchronizer.frames_processed_event.clear()

            if not waited:
                if self.frame_synchronizer.is_all_completed():
                    print("所有视频处理完成")
                    self.is_running = False
                    break
                if last_combined_view is None:
                    try:
                        map_frame = self.tunnel_map.render(time.time())
                        last_combined_view = self.view_composer.create_combined_view({}, map_frame, time.time())
                    except Exception as e:
                        print(f"渲染等待画面失败: {e}")
                if last_combined_view is not None:
                    cv2.imshow(window_name, last_combined_view)
                self._handle_keyboard_input()
                continue

            time_point, processed_results = self.frame_synchronizer.get_latest_processed_results()
            if time_point is None or not processed_results:
                if last_combined_view is None:
                    try:
                        map_frame = self.tunnel_map.render(time.time())
                        last_combined_view = self.view_composer.create_combined_view({}, map_frame, time.time())
                    except Exception as e:
                        print(f"渲染等待画面失败: {e}")
                if last_combined_view is not None:
                    cv2.imshow(window_name, last_combined_view)
                self._handle_keyboard_input()
                continue

            current_time = time.time()
            elapsed = current_time - last_display_time
            target_time = target_frame_time / self.speed_factor

            if elapsed < target_time:
                time.sleep(max(0.001, target_time - elapsed))

            last_display_time = time.time()

            camera_frames = {}
            for cam_id, result in processed_results.items():
                if result.processed_frame is not None:
                    camera_frames[cam_id] = result.processed_frame
                    self.output_manager.write_camera_video(cam_id, result.processed_frame)

            map_frame = self.tunnel_map.render(time_point)

            combined_view = self.view_composer.create_combined_view(
                camera_frames, map_frame, time_point
            )
            last_combined_view = combined_view
            cv2.imshow(window_name, combined_view)
            if camera_frames:
                self.output_manager.write_integrated_video(combined_view)

            self._handle_keyboard_input()
    
    def _handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.stop()
        elif key == ord('+') or key == ord('='):
            self.speed_factor = min(4.0, self.speed_factor + 0.25)
            print(f"播放速度: {self.speed_factor:.1f}x")
        elif key == ord('-') or key == ord('_'):
            self.speed_factor = max(0.25, self.speed_factor - 0.25)
            print(f"播放速度: {self.speed_factor:.1f}x")
    
    def stop(self):
        self.is_running = False

        if self.ws_push_manager:
            self.ws_push_manager.stop()

        for thread in self.capture_threads.values():
            thread.join(timeout=1.0)

        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.display_thread:
            self.display_thread.join(timeout=1.0)

        self.video_capture.close_videos()
        self.output_manager.close_all_writers()
        cv2.destroyAllWindows()

        self._print_statistics()
    
    def _print_statistics(self):
        if not self.statistics.start_time:
            return
        
        end_time = time.time()
        total_time = end_time - self.statistics.start_time
        
        print(f"\n=== 系统性能统计 ===")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"总处理帧数: {self.statistics.total_frames}")
        print(f"平均处理速率: {self.statistics.total_frames / total_time:.2f} FPS")
        
        print(f"\n=== 摄像头统计 ===")
        for cam_id, fps in self.processing_fps.items():
            times = self.processing_times.get(cam_id, [])
            avg_time = sum(times) / len(times) if times else 0
            print(f"{cam_id}: {fps:.2f} FPS, 平均处理时间: {avg_time*1000:.2f} ms")
        
        print(f"\n=== 车辆统计 ===")
        print(f"处理车辆总数: {self.statistics.processed_vehicles}")
        print(f"检测区域外车辆: {self.statistics.filtered_vehicles}")
        
        tracked_vehicles = len(self.id_manager.prediction_manager.vehicle_tracking_info)
        predicted_vehicles = len(self.id_manager.prediction_manager.predictors)
        print(f"当前跟踪车辆: {tracked_vehicles}")
        print(f"预测中车辆: {predicted_vehicles}")
        
        buffer_count = len(self.id_manager.vehicle_buffer)
        print(f"活跃缓冲区: {buffer_count}")
        
        print(f"\n=== 系统信息 ===")
        print(f"隧道实际尺寸: {config.TUNNEL_REAL_DIMENSIONS['length']}m × {config.TUNNEL_REAL_DIMENSIONS['width']}m")
        print(f"地图像素尺寸: {config.MAP_WIDTH} × {config.MAP_HEIGHT} 像素")
        print(f"坐标分辨率: {config.COORDINATE_CALIBRATION['meters_per_pixel_x']:.3f} m/像素")
        print(f"图像缓冲区大小: {self.id_manager.buffer_size} 帧")
        print(f"缓冲区超时时间: {self.id_manager.buffer_timeout} 秒")
    
    def process_video_files(self, output_video: bool = True):
        self.start(output_video)
        
        try:
            self._run_display_loop()
        except KeyboardInterrupt:
            print("收到键盘中断，正在停止...")
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="增强版车辆跟踪系统 - 支持丢失车辆预测和图像缓冲")
    parser.add_argument("--no-output", action="store_true", help="不保存输出视频")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度")
    parser.add_argument("--width", type=int, default=2000, help="窗口宽度")
    parser.add_argument("--height", type=int, default=1000, help="窗口高度")
    args = parser.parse_args()
    
    system = EnhancedVehicleSystem(output_width=args.width, output_height=args.height)
    system.speed_factor = args.speed
    
    system.process_video_files(output_video=not args.no_output)


if __name__ == "__main__":
    main()
