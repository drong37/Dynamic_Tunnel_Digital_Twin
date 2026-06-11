
import cv2
import numpy as np
import time
import threading
import logging
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Deque

import config
from lane_classifier import LaneClassifier
from vehicle_speed_calculator import VehicleSpeedCalculator

logger = logging.getLogger("VehiclePredictionManager")


class VehicleState(Enum):
    DETECTED = "detected"
    LOST = "lost"


class MovementDirection(Enum):
    UNKNOWN = "unknown"
    LEFT = "left"
    RIGHT = "right"
    STATIONARY = "stationary"


@dataclass
class VehicleMotionParams:
    speed: Tuple[float, float] = (0.0, 0.0)
    direction: float = 0.0
    movement_direction: MovementDirection = MovementDirection.UNKNOWN
    direction_locked: bool = False
    direction_confidence: float = 0.0
    
    last_position: Tuple[int, int] = (0, 0)
    lane_y: float = 50.0
    last_camera: str = ""
    
    enter_lost_time: Optional[float] = None
    
    direction_history: List[float] = field(default_factory=list)
    position_history: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class VehicleTrackingInfo:
    global_id: int
    last_detection_time: float
    last_position: Tuple[int, int]
    last_camera: str
    last_bbox: List[float]
    detection_count: int = 0
    consecutive_misses: int = 0


class VehicleStateManager:
    
    def __init__(self):
        self.vehicle_states: Dict[int, VehicleState] = {}
        self.motion_params: Dict[int, VehicleMotionParams] = {}
        
        self.direction_lock_threshold = 3
        self.direction_confidence_threshold = 0.8
        self.min_movement_distance = 5
    
    def get_state(self, global_id: int) -> VehicleState:
        return self.vehicle_states.get(global_id, VehicleState.LOST)
    
    def set_state(self, global_id: int, state: VehicleState):
        old_state = self.vehicle_states.get(global_id, VehicleState.LOST)
        self.vehicle_states[global_id] = state
        
        if old_state != state:
            logger.info(f"车辆 {global_id} 状态变更: {old_state.value if old_state else 'None'} -> {state.value}")
            
            if state == VehicleState.LOST:
                if global_id not in self.motion_params:
                    self.motion_params[global_id] = VehicleMotionParams()
                self.motion_params[global_id].enter_lost_time = time.time()
    
    def update_motion_params(self, global_id: int, current_pos: Tuple[int, int], 
                           last_pos: Tuple[int, int], time_diff: float, lane_y: float):
        if global_id not in self.motion_params:
            self.motion_params[global_id] = VehicleMotionParams()
        
        params = self.motion_params[global_id]
        
        params.position_history.append(current_pos)
        if len(params.position_history) > 10:
            params.position_history = params.position_history[-10:]
        
        if time_diff > 0:
            dx = current_pos[0] - last_pos[0]
            dy = current_pos[1] - last_pos[1]
            
            speed_x = dx / time_diff
            speed_y = dy / time_diff
            params.speed = (speed_x, speed_y)
            
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                params.direction = np.arctan2(dy, dx)
                params.direction_history.append(params.direction)
                if len(params.direction_history) > 10:
                    params.direction_history = params.direction_history[-10:]
            
            self._update_movement_direction(params, dx, dy)
        
        params.last_position = current_pos
        params.lane_y = lane_y
        
        logger.debug(f"车辆 {global_id} 运动参数更新: 速度=({params.speed[0]:.2f}, {params.speed[1]:.2f}) px/s, 方向={params.movement_direction.value}")
    
    def _update_movement_direction(self, params: VehicleMotionParams, dx: float, dy: float):
        if params.direction_locked:
            return
        
        if abs(dx) < self.min_movement_distance:
            return
        
        if dx > 0:
            current_direction = MovementDirection.RIGHT
        elif dx < 0:
            current_direction = MovementDirection.LEFT
        else:
            current_direction = MovementDirection.STATIONARY
        
        if params.movement_direction == MovementDirection.UNKNOWN:
            params.movement_direction = current_direction
            params.direction_confidence = 1.0
            logger.info(f"车辆方向初始化为: {current_direction.value}")
            return
        
        if params.movement_direction == current_direction:
            params.direction_confidence = min(1.0, params.direction_confidence + 0.2)
            
            if params.direction_confidence >= self.direction_confidence_threshold:
                params.direction_locked = True
                logger.info(f"车辆方向已锁定为: {current_direction.value}")
        else:
            params.direction_confidence = max(0.0, params.direction_confidence - 0.3)
            
            if params.direction_confidence < 0.3:
                params.movement_direction = current_direction
                params.direction_confidence = 0.5
                logger.info(f"车辆方向更新为: {current_direction.value}")
    
    def get_movement_direction(self, global_id: int) -> MovementDirection:
        if global_id not in self.motion_params:
            return MovementDirection.UNKNOWN
        return self.motion_params[global_id].movement_direction
    
    def is_direction_locked(self, global_id: int) -> bool:
        if global_id not in self.motion_params:
            return False
        return self.motion_params[global_id].direction_locked
    
    def get_motion_params(self, global_id: int) -> Tuple[Tuple[float, float], float, Tuple[int, int], float]:
        params = self.motion_params.get(global_id, VehicleMotionParams())
        return params.speed, params.direction, params.last_position, params.lane_y
    
    def should_start_blind_zone_prediction(self, global_id: int, current_camera: str, 
                                         position: Tuple[int, int]) -> bool:
        if global_id not in self.motion_params:
            return False
        
        params = self.motion_params[global_id]
        
        if params.movement_direction == MovementDirection.UNKNOWN:
            return False
        
        return self._is_near_camera_boundary(current_camera, position, params.movement_direction)
    
    def _is_near_camera_boundary(self, camera_id: str, position: Tuple[int, int], 
                               direction: MovementDirection) -> bool:
        try:
            camera_region = config.CAMERA_CONFIG[camera_id]['calibration']['map_region']
            x_min, _, x_max, _ = camera_region
            x, _ = position
            
            boundary_threshold = 50
            
            if direction == MovementDirection.RIGHT:
                return x >= (x_max - boundary_threshold)
            elif direction == MovementDirection.LEFT:
                return x <= (x_min + boundary_threshold)
            
            return False
        except Exception:
            return False
    
    def set_enter_lost(self, global_id: int, timestamp: float):
        if global_id not in self.motion_params:
            self.motion_params[global_id] = VehicleMotionParams()
        self.motion_params[global_id].enter_lost_time = timestamp
        logger.info(f"车辆 {global_id} 进入丢失状态，时间: {timestamp:.2f}s")
    
    def set_enter_blind_zone(self, global_id: int, timestamp: float):
        self.set_enter_lost(global_id, timestamp)
    
    def set_enter_coverage_loss(self, global_id: int, timestamp: float):
        self.set_enter_lost(global_id, timestamp)
    
    def cleanup_lost_vehicles(self, current_time: float, timeout: float = 60.0):
        vehicles_to_remove = []
        
        for global_id, state in list(self.vehicle_states.items()):
            if state == VehicleState.LOST:
                params = self.motion_params.get(global_id)
                if params and params.enter_lost_time:
                    if current_time - params.enter_lost_time > timeout:
                        vehicles_to_remove.append(global_id)
        
        for global_id in vehicles_to_remove:
            self._remove_vehicle(global_id)
            logger.info(f"清理超时车辆 {global_id}")
    
    def _remove_vehicle(self, global_id: int):
        self.vehicle_states.pop(global_id, None)
        self.motion_params.pop(global_id, None)


class VehiclePredictor:
    
    def __init__(self, global_id: int, initial_position: Tuple[int, int], 
                 movement_direction: MovementDirection, speed: Tuple[float, float], 
                 lane_y: float, start_time: float, current_camera: str, 
                 prediction_type: str = "lost"):
        self.global_id = global_id
        self.start_position = initial_position
        self.movement_direction = movement_direction
        self.speed_x, self.speed_y = speed
        self.lane_y = lane_y
        self.start_time = start_time
        self.current_camera = current_camera
        self.prediction_type = prediction_type
        self.is_active = True
        
        if self.movement_direction == MovementDirection.RIGHT:
            self.speed_x = abs(self.speed_x) if self.speed_x != 0 else 2.0
        elif self.movement_direction == MovementDirection.LEFT:
            self.speed_x = -abs(self.speed_x) if self.speed_x != 0 else -2.0
        else:
            if abs(self.speed_x) < 1.0:
                self.speed_x = 2.0 if self.speed_x >= 0 else -2.0
        
        self.speed_x = max(-8.0, min(8.0, self.speed_x))
        self.speed_y = max(-2.0, min(2.0, self.speed_y))
        
        self.max_prediction_time = float(config.TRACKING_STRATEGY['time_windows']['max_prediction_time_sec'])
        self.speed_decay_factor = 0.99
        
        self.prediction_boundary = self._calculate_prediction_boundary()
        
        logger.info(f"启动车辆 {global_id} {prediction_type} 预测:")
        logger.info(f"  起始位置: {initial_position}")
        logger.info(f"  运动方向: {movement_direction.value}")
        logger.info(f"  速度: ({self.speed_x:.2f}, {self.speed_y:.2f}) px/s")
        logger.info(f"  预测边界: {self.prediction_boundary}")
    
    def _calculate_prediction_boundary(self) -> Optional[int]:
        try:
            if self.movement_direction == MovementDirection.RIGHT:
                next_camera = self._get_next_camera_right(self.current_camera)
                if next_camera:
                    next_region = config.CAMERA_CONFIG[next_camera]['calibration']['map_region']
                    next_left, _, _, _ = next_region
                    return next_left + 50
                else:
                    return config.MAP_WIDTH
            
            elif self.movement_direction == MovementDirection.LEFT:
                next_camera = self._get_next_camera_left(self.current_camera)
                if next_camera:
                    next_region = config.CAMERA_CONFIG[next_camera]['calibration']['map_region']
                    _, _, next_right, _ = next_region
                    return next_right - 50
                else:
                    return 50
            
            prediction_distance = 200
            if self.speed_x > 0:
                return min(config.MAP_WIDTH - 50, self.start_position[0] + prediction_distance)
            else:
                return max(50, self.start_position[0] - prediction_distance)
                
        except Exception as e:
            logger.error(f"计算预测边界失败: {e}")
            if self.speed_x > 0:
                return min(config.MAP_WIDTH - 50, self.start_position[0] + 150)
            else:
                return max(50, self.start_position[0] - 150)
    
    def _get_next_camera_right(self, current_camera: str) -> Optional[str]:
        camera_order = config.camera_manager.camera_config.get_camera_order()
        try:
            current_index = camera_order.index(current_camera)
            if current_index < len(camera_order) - 1:
                return camera_order[current_index + 1]
        except ValueError:
            logger.warning(f"摄像头 {current_camera} 不在配置的摄像头顺序中")
        return None
    
    def _get_next_camera_left(self, current_camera: str) -> Optional[str]:
        camera_order = config.camera_manager.camera_config.get_camera_order()
        try:
            current_index = camera_order.index(current_camera)
            if current_index > 0:
                return camera_order[current_index - 1]
        except ValueError:
            logger.warning(f"摄像头 {current_camera} 不在配置的摄像头顺序中")
        return None
    
    def predict_position(self, current_time: float) -> Optional[Tuple[int, int]]:
        if not self.is_active:
            return None
        
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > self.max_prediction_time:
            logger.info(f"车辆 {self.global_id} 预测超过最大时间 {self.max_prediction_time}s，终止预测")
            self.is_active = False
            return None
        
        decay_factor = self.speed_decay_factor ** (elapsed_time / 3.0)
        current_speed_x = self.speed_x * decay_factor
        
        predicted_x = self.start_position[0] + current_speed_x * elapsed_time
        predicted_y = self.lane_y
        
        predicted_x = max(0, min(config.MAP_WIDTH - 1, predicted_x))
        predicted_y = max(0, min(config.MAP_HEIGHT - 1, predicted_y))
        
        if self.prediction_boundary is not None:
            if self.movement_direction == MovementDirection.RIGHT:
                if predicted_x >= self.prediction_boundary:
                    logger.info(f"车辆 {self.global_id} 预测到达右边界 {self.prediction_boundary}，终止预测")
                    self.is_active = False
                    return None
            elif self.movement_direction == MovementDirection.LEFT:
                if predicted_x <= self.prediction_boundary:
                    logger.info(f"车辆 {self.global_id} 预测到达左边界 {self.prediction_boundary}，终止预测")
                    self.is_active = False
                    return None
        
        return (int(predicted_x), int(predicted_y))
    
    def get_speed_kmh(self) -> float:
        total_speed = abs(self.speed_x)
        if total_speed <= 0:
            return 0
        speed_mps = total_speed * config.COORDINATE_CALIBRATION['meters_per_pixel_x']
        return speed_mps * 3.6


class VehiclePredictionManager:
    
    def __init__(self, tunnel_map=None):
        self.state_manager = VehicleStateManager()
        self.tunnel_map = tunnel_map
        
        self.predictors: Dict[int, VehiclePredictor] = {}
        
        self.vehicle_tracking_info: Dict[int, VehicleTrackingInfo] = {}

        windows = config.TRACKING_STRATEGY['time_windows']
        suppression = config.TRACKING_STRATEGY['suppression']
        
        self.lost_vehicle_timeout = float(windows['lost_vehicle_timeout_sec'])
        self.prediction_start_delay = float(windows['prediction_start_delay_sec'])
        self.prediction_timeout = float(windows['prediction_timeout_sec'])
        self.prediction_update_interval = float(windows['prediction_update_interval_sec'])
        self.last_prediction_update = 0
        
        self.spatial_suppression_base = float(suppression['base_pixels'])
        self.spatial_suppression_min = float(suppression['min_pixels'])
        self.spatial_suppression_max = float(suppression['max_pixels'])
        self.suppression_time_horizon = float(suppression['time_horizon_sec'])
        
        self.lock = threading.RLock()
        
        logger.info("车辆预测管理器已初始化")

    def _get_adaptive_suppression_threshold(self, predictor: 'VehiclePredictor') -> float:
        speed_px_s = float((predictor.speed_x ** 2 + predictor.speed_y ** 2) ** 0.5)
        adaptive = self.spatial_suppression_base + speed_px_s * self.suppression_time_horizon
        return max(self.spatial_suppression_min, min(self.spatial_suppression_max, adaptive))

    def _check_spatial_overlap_and_suppress(self, detected_position: Tuple[int, int], exclude_id: int):
        if detected_position is None or len(detected_position) != 2:
            logger.warning("空间抑制收到无效位置，跳过处理")
            return

        predictors_to_remove = []
        x1, y1 = detected_position

        if not np.isfinite(x1) or not np.isfinite(y1):
            logger.warning(f"空间抑制收到非有限坐标: {detected_position}")
            return
        
        for global_id, predictor in self.predictors.items():
            if global_id == exclude_id:
                continue
                
            pred_x, pred_y = predictor.start_position
            if hasattr(predictor, 'last_predicted_pos') and predictor.last_predicted_pos:
                pred_x, pred_y = predictor.last_predicted_pos
            
            distance = ((x1 - pred_x) ** 2 + (y1 - pred_y) ** 2) ** 0.5
            adaptive_threshold = self._get_adaptive_suppression_threshold(predictor)

            if distance < adaptive_threshold:
                predictors_to_remove.append(global_id)
                logger.info(
                    f"检测到ID切换: 实测车辆 {exclude_id} 与 预测车辆 {global_id} 重叠 "
                    f"(距离 {distance:.1f} < 阈值 {adaptive_threshold:.1f})，移除旧预测"
                )
        
        for global_id in predictors_to_remove:
            self.predictors.pop(global_id, None)
            self.state_manager._remove_vehicle(global_id)
            if global_id in self.vehicle_tracking_info:
                del self.vehicle_tracking_info[global_id]
    
    def update_vehicle_tracking_info(self, global_id: int, camera_id: str, 
                                   bbox: List[float], time_point: float):
        if global_id not in self.vehicle_tracking_info:
            self.vehicle_tracking_info[global_id] = VehicleTrackingInfo(
                global_id=global_id,
                last_detection_time=time_point,
                last_position=(int((bbox[0] + bbox[2]) / 2), int(bbox[3])),
                last_camera=camera_id,
                last_bbox=bbox
            )
        
        info = self.vehicle_tracking_info[global_id]
        info.last_detection_time = time_point
        info.last_position = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
        info.last_camera = camera_id
        info.last_bbox = bbox
        info.detection_count += 1
        info.consecutive_misses = 0
    
    def handle_lost_vehicles(self, current_time: float, detected_global_ids: set):
        with self.lock:
            for global_id, info in list(self.vehicle_tracking_info.items()):
                if global_id not in detected_global_ids:
                    info.consecutive_misses += 1
                    time_since_last_detection = current_time - info.last_detection_time
                    
                    if (time_since_last_detection >= self.prediction_start_delay and 
                        time_since_last_detection <= self.lost_vehicle_timeout and
                        info.detection_count >= 2):
                        
                        self._start_lost_vehicle_prediction(global_id, info, current_time)
                    
                    elif time_since_last_detection > self.lost_vehicle_timeout:
                        self._cleanup_lost_vehicle(global_id)
    
    def _start_lost_vehicle_prediction(self, global_id: int, info: VehicleTrackingInfo, current_time: float):
        current_state = self.state_manager.get_state(global_id)
        if current_state == VehicleState.LOST and global_id in self.predictors:
            return
        
        speed, direction, last_position, lane_y = self.state_manager.get_motion_params(global_id)
        
        if abs(speed[0]) < 0.1 and abs(speed[1]) < 0.1:
            estimated_speed = self._estimate_vehicle_speed(global_id, info)
            if estimated_speed is not None:
                speed = estimated_speed
            else:
                return
        
        map_position = self._get_vehicle_map_position(global_id, info)
        if map_position is None:
            return
        
        try:
            success = self.start_prediction(
                global_id=global_id,
                last_position=map_position,
                speed=speed,
                direction=direction,
                lane_y=lane_y,
                start_time=current_time,
                last_camera=info.last_camera
            )
            
            if success:
                logger.info(f"为丢失车辆 {global_id} 启动预测，最后在摄像头 {info.last_camera}")
            
        except Exception as e:
            logger.error(f"启动丢失车辆预测失败: {e}")
    
    def _estimate_vehicle_speed(self, global_id: int, info: VehicleTrackingInfo) -> Optional[Tuple[float, float]]:
        if not self.tunnel_map:
            return None
        
        if global_id in self.tunnel_map.trajectory_manager.global_trajectories:
            trajectory_data = self.tunnel_map.trajectory_manager.global_trajectories[global_id]
            trajectory = trajectory_data.trajectory
            
            if len(trajectory) >= 3:
                recent_positions = list(trajectory)[-3:]
                
                total_dx, total_dy = 0, 0
                segments = len(recent_positions) - 1
                
                for i in range(segments):
                    pos1, pos2 = recent_positions[i], recent_positions[i + 1]
                    total_dx += (pos2[0] - pos1[0])
                    total_dy += (pos2[1] - pos1[1])
                
                time_interval = segments * (1.0 / config.TARGET_FPS)
                speed_x = total_dx / time_interval if time_interval > 0 else 0
                speed_y = total_dy / time_interval if time_interval > 0 else 0
                
                if abs(speed_x) < 1.0:
                    overall_dx = recent_positions[-1][0] - recent_positions[0][0]
                    
                    if global_id in self.state_manager.motion_params:
                        prev_speed = self.state_manager.motion_params[global_id].speed
                        if abs(prev_speed[0]) > 0.8:
                            speed_x = prev_speed[0] * 0.9
                        elif abs(overall_dx) > 15:
                            direction = 1 if overall_dx > 0 else -1
                            speed_x = direction * 1.5
                        else:
                            if prev_speed[0] != 0:
                                speed_x = (1 if prev_speed[0] > 0 else -1) * 1.0
                            else:
                                speed_x = 1.5
                    elif abs(overall_dx) > 15:
                        direction = 1 if overall_dx > 0 else -1
                        speed_x = direction * 1.5
                    else:
                        speed_x = 1.5
                
                speed_x = max(-8.0, min(8.0, speed_x))
                speed_y = max(-2.0, min(2.0, speed_y))
                
                return (speed_x, speed_y)
        
        if global_id in self.state_manager.motion_params:
            prev_speed = self.state_manager.motion_params[global_id].speed
            if abs(prev_speed[0]) > 0.5:
                return prev_speed
        
        try:
            camera_order = config.camera_manager.camera_config.get_camera_order()
            cam_index = camera_order.index(info.last_camera)
            if cam_index < len(camera_order) - 1:
                return (3.0, 0.0)
            else:
                return (-3.0, 0.0)
        except ValueError:
            return (3.0, 0.0)
    
    def _get_vehicle_map_position(self, global_id: int, info: VehicleTrackingInfo) -> Optional[Tuple[int, int]]:
        if not self.tunnel_map:
            return None
        
        if global_id in self.tunnel_map.trajectory_manager.global_trajectories:
            trajectory_data = self.tunnel_map.trajectory_manager.global_trajectories[global_id]
            if len(trajectory_data.trajectory) > 0:
                return trajectory_data.trajectory[-1]
        
        try:
            camera_id = info.last_camera
            bbox = info.last_bbox
            
            if camera_id in self.tunnel_map.lane_classifiers:
                lane_classifier = self.tunnel_map.lane_classifiers[camera_id]
                lane_index, lane_name = lane_classifier.determine_lane(bbox)
                
                if camera_id in self.tunnel_map.camera_calibrations:
                    bottom_center = [int((bbox[0] + bbox[2]) / 2), int(bbox[3])]
                    map_position = self.tunnel_map.camera_calibrations[camera_id].map_to_ground(bottom_center)
                    
                    map_lane_index = self.tunnel_map.lane_mapping[camera_id].get(lane_index, 0)
                    if map_lane_index < len(config.MAP_LANE_DEFINITIONS):
                        map_lane = config.MAP_LANE_DEFINITIONS[map_lane_index]
                        lane_y_min, lane_y_max = map_lane["y_range"]
                        y = (lane_y_min + lane_y_max) / 2
                        
                        return (int(map_position[0]), int(y))
        except Exception as e:
            logger.error(f"获取车辆地图位置失败: {e}")
        
        return None
    
    def _cleanup_lost_vehicle(self, global_id: int):
        if global_id in self.vehicle_tracking_info:
            del self.vehicle_tracking_info[global_id]
            logger.info(f"清理长时间丢失的车辆 {global_id}")
    
    def start_prediction(self, global_id: int, last_position: Tuple[int, int],
                        speed: Tuple[float, float], direction: float, 
                        lane_y: float, start_time: float, last_camera: str) -> bool:
        try:
            if global_id in self.predictors:
                return False
            
            with self.lock:
                movement_direction = self.state_manager.get_movement_direction(global_id)
                
                if movement_direction == MovementDirection.UNKNOWN:
                    if speed[0] > 0:
                        movement_direction = MovementDirection.RIGHT
                    elif speed[0] < 0:
                        movement_direction = MovementDirection.LEFT
                    else:
                        movement_direction = MovementDirection.STATIONARY
                
                actual_lane_y = last_position[1]
                
                predictor = VehiclePredictor(
                    global_id=global_id,
                    initial_position=last_position,
                    movement_direction=movement_direction,
                    speed=speed,
                    lane_y=actual_lane_y,
                    start_time=start_time,
                    current_camera=last_camera,
                    prediction_type="lost"
                )
                
                self.predictors[global_id] = predictor
                self.state_manager.set_state(global_id, VehicleState.LOST)
                self.state_manager.set_enter_lost(global_id, start_time)
                
                logger.info(f"启动车辆 {global_id} 预测，最后位置: {last_position}，方向: {movement_direction.value}")
                return True
                
        except Exception as e:
            logger.error(f"启动车辆预测失败: {e}")
            return False
    
    def update_predictions(self, current_time: float) -> Dict[int, Tuple[int, int]]:
        with self.lock:
            time_since_last = current_time - self.last_prediction_update
            if time_since_last < self.prediction_update_interval:
                return {}
            
            self.last_prediction_update = current_time
            
            predicted_positions = {}
            predictors_to_remove = []
            
            for global_id, predictor in list(self.predictors.items()):
                if not predictor.is_active:
                    predictors_to_remove.append(global_id)
                    continue
                
                predicted_position = predictor.predict_position(current_time)
                if predicted_position is not None:
                    predicted_positions[global_id] = predicted_position
                    predictor.last_predicted_pos = predicted_position
                else:
                    predictors_to_remove.append(global_id)
            
            for global_id in predictors_to_remove:
                self.predictors.pop(global_id, None)
                current_state = self.state_manager.get_state(global_id)
                if current_state == VehicleState.LOST:
                    pass
            
            return predicted_positions
    
    def handle_vehicle_detected(self, global_id: int, camera_id: str, position: Tuple[int, int], 
                               current_time: float, was_in_prediction: bool = False):
        with self.lock:
            if was_in_prediction:
                logger.info(f"车辆 {global_id} 从预测状态重新被摄像头 {camera_id} 检测到")
                self.predictors.pop(global_id, None)
                self.state_manager.set_state(global_id, VehicleState.DETECTED)
            
            self._check_spatial_overlap_and_suppress(position, global_id)
    
    def cleanup_expired_predictions(self, current_time: float):
        with self.lock:
            expired_predictors = [
                global_id for global_id, predictor in self.predictors.items()
                if current_time - predictor.start_time > self.prediction_timeout
            ]
            
            for global_id in expired_predictors:
                logger.debug(f"清理超时预测器: 车辆 {global_id}")
                self.predictors.pop(global_id, None)
            
            self.state_manager.cleanup_lost_vehicles(current_time, self.prediction_timeout)
    
    def get_predictor_info(self, global_id: int) -> Optional[Dict]:
        if global_id in self.predictors:
            predictor = self.predictors[global_id]
            return {
                'global_id': predictor.global_id,
                'movement_direction': predictor.movement_direction.value,
                'speed': (predictor.speed_x, predictor.speed_y),
                'speed_kmh': predictor.get_speed_kmh(),
                'prediction_type': predictor.prediction_type,
                'start_time': predictor.start_time,
                'is_active': predictor.is_active
            }
        return None
    
    def get_all_predictors_info(self) -> Dict[int, Dict]:
        return {global_id: info for global_id in self.predictors.keys() 
                if (info := self.get_predictor_info(global_id)) is not None}
    
    def is_vehicle_in_prediction(self, global_id: int) -> bool:
        return global_id in self.predictors
    
    def get_vehicle_state(self, global_id: int) -> VehicleState:
        return self.state_manager.get_state(global_id)
    
    def set_vehicle_state(self, global_id: int, state: VehicleState):
        self.state_manager.set_state(global_id, state)
    
    def update_motion_params(self, global_id: int, current_pos: Tuple[int, int], 
                           last_pos: Tuple[int, int], time_diff: float, lane_y: float):
        self.state_manager.update_motion_params(global_id, current_pos, last_pos, time_diff, lane_y)
    
    def get_motion_params(self, global_id: int) -> Tuple[Tuple[float, float], float, Tuple[int, int], float]:
        return self.state_manager.get_motion_params(global_id)
    
    def should_start_blind_zone_prediction(self, global_id: int, current_camera: str, 
                                         position: Tuple[int, int]) -> bool:
        return self.state_manager.should_start_blind_zone_prediction(global_id, current_camera, position)
    
    def get_movement_direction(self, global_id: int) -> MovementDirection:
        return self.state_manager.get_movement_direction(global_id)
    
    def get_statistics(self) -> Dict:
        return {
            'active_predictors': len(self.predictors),
            'tracked_vehicles': len(self.vehicle_tracking_info),
            'vehicle_states': {
                'detected': len([v for v in self.state_manager.vehicle_states.values() if v == VehicleState.DETECTED]),
                'lost': len([v for v in self.state_manager.vehicle_states.values() if v == VehicleState.LOST])
            }
        }