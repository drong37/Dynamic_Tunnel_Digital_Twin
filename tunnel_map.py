
import cv2
import numpy as np
import time
import threading
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Deque, Any
import logging

import config
from lane_classifier import LaneClassifier
from vehicle_speed_calculator import VehicleSpeedCalculator
from vehicle_prediction_manager import VehiclePredictionManager, VehicleState, MovementDirection, VehiclePredictor

logger = logging.getLogger("TunnelMap")






@dataclass
class TrajectoryData:
    trajectory: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=20))
    color: Tuple[int, int, int] = (255, 0, 0)
    last_update: float = 0.0
    info: Dict = field(default_factory=dict)
    
    last_real_position: Optional[Tuple[int, int]] = None
    last_real_update: float = 0.0
    
    movement_direction: MovementDirection = MovementDirection.UNKNOWN
    direction_locked: bool = False


class DrawingUtils:
    
    @staticmethod
    def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                        color: Tuple[int, int, int], thickness: int = 1, 
                        dash_length: int = 10, gap_length: int = 10):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        if dist == 0:
            return
        
        dashes = int(dist / (dash_length + gap_length))
        ratio = dash_length / (dash_length + gap_length)
        
        for i in range(dashes):
            start_ratio = i / dashes
            end_ratio = min((i + ratio) / dashes, 1.0)
            
            start_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                       int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
            end_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                     int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
            
            cv2.line(img, start_pt, end_pt, color, thickness)
    
    @staticmethod
    def get_color_by_id(idx: int) -> Tuple[int, int, int]:
        idx = abs(int(idx)) % 256
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        if idx < len(colors):
            return colors[idx]
        return (idx * 33 % 256, idx * 73 % 256, idx * 123 % 256)






class VehicleTrajectoryManager:
    
    def __init__(self, max_trajectory_length: int = 20, trajectory_timeout: float = 10.0):
        self.max_trajectory_length = max_trajectory_length
        self.trajectory_timeout = trajectory_timeout
        
        self.global_trajectories: Dict[int, TrajectoryData] = {}
        self.global_speeds: Dict[int, float] = {}
        self.global_lanes: Dict[int, float] = {}
        
        self.local_vehicle_info: Dict[Tuple[str, int], Dict] = {}
        
        self.position_smoothing_factor = 0.3
        self.max_position_jump = 30
    
    def update_local_trajectory(self, camera_id: str, global_id: int, position: Tuple[int, int], 
                              current_time: float, vehicle_info: Dict):
        key = (camera_id, global_id)
        self.local_vehicle_info[key] = {
            'position': position,
            'timestamp': current_time,
            'info': vehicle_info
        }
    
    def update_global_trajectory(self, global_id: int, position: Tuple[int, int], 
                               current_time: float, camera_id: str, lane_y: float,
                               movement_direction: MovementDirection = MovementDirection.UNKNOWN):
        if global_id not in self.global_trajectories:
            self.global_trajectories[global_id] = TrajectoryData(
                trajectory=deque(maxlen=self.max_trajectory_length),
                color=DrawingUtils.get_color_by_id(hash(str(global_id)) % 1000),
                last_update=current_time,
                movement_direction=movement_direction
            )
        
        trajectory_data = self.global_trajectories[global_id]
        
        smoothed_position = self._smooth_position(global_id, position, trajectory_data)
        
        if not trajectory_data.direction_locked:
            self._update_trajectory_direction(trajectory_data, smoothed_position, movement_direction)
        
        trajectory_data.trajectory.append(smoothed_position)
        trajectory_data.last_update = current_time
        
        trajectory_data.last_real_position = smoothed_position
        trajectory_data.last_real_update = current_time
        
        self.global_lanes[global_id] = smoothed_position[1]
        
        logger.debug(f"更新车辆 {global_id} 轨迹: {smoothed_position}, 方向: {trajectory_data.movement_direction.value}")
    
    def _smooth_position(self, global_id: int, new_position: Tuple[int, int], 
                        trajectory_data: TrajectoryData) -> Tuple[int, int]:
        if len(trajectory_data.trajectory) == 0:
            return self._clamp_to_map_bounds(new_position)
        
        last_position = trajectory_data.trajectory[-1]
        distance = ((new_position[0] - last_position[0]) ** 2 + 
                   (new_position[1] - last_position[1]) ** 2) ** 0.5
        
        if distance > self.max_position_jump:
            time_diff = trajectory_data.last_update - (trajectory_data.last_real_update or 0)
            if time_diff > 0:
                speed = distance / time_diff
                if speed > 100:
                    logger.warning(f"车辆 {global_id} 位置跳跃过大: {last_position} -> {new_position}, "
                                 f"距离: {distance:.1f}, 速度: {speed:.1f} px/s")
                    
                    alpha = self.position_smoothing_factor
                    smoothed_x = int(last_position[0] * (1 - alpha) + new_position[0] * alpha)
                    smoothed_y = int(last_position[1] * (1 - alpha) + new_position[1] * alpha)
                    return self._clamp_to_map_bounds((smoothed_x, smoothed_y))
        
        if len(trajectory_data.trajectory) >= 2:
            alpha = 0.7
            smoothed_x = int(last_position[0] * (1 - alpha) + new_position[0] * alpha)
            smoothed_y = int(last_position[1] * (1 - alpha) + new_position[1] * alpha)
            return self._clamp_to_map_bounds((smoothed_x, smoothed_y))
        
        return self._clamp_to_map_bounds(new_position)
    
    def _clamp_to_map_bounds(self, position: Tuple[int, int]) -> Tuple[int, int]:
        x, y = position
        
        max_x = config.MAP_WIDTH - 1
        max_y = config.MAP_HEIGHT - 1
        
        clamped_x = max(0, min(max_x, x))
        clamped_y = max(0, min(max_y, y))
        
        return (int(clamped_x), int(clamped_y))
    
    def _update_trajectory_direction(self, trajectory_data: TrajectoryData, 
                                   new_position: Tuple[int, int], 
                                   detected_direction: MovementDirection):
        if trajectory_data.direction_locked:
            return
        
        if len(trajectory_data.trajectory) >= 3:
            recent_positions = list(trajectory_data.trajectory)[-3:] + [new_position]
            
            total_dx = recent_positions[-1][0] - recent_positions[0][0]
            
            if abs(total_dx) > 10:
                inferred_direction = MovementDirection.RIGHT if total_dx > 0 else MovementDirection.LEFT
                
                if detected_direction == MovementDirection.UNKNOWN or detected_direction == inferred_direction:
                    trajectory_data.movement_direction = inferred_direction
                    trajectory_data.direction_locked = True
                    logger.info(f"轨迹方向已锁定为: {inferred_direction.value}")
                elif detected_direction != inferred_direction:
                    if trajectory_data.movement_direction == MovementDirection.UNKNOWN:
                        trajectory_data.movement_direction = detected_direction
        elif detected_direction != MovementDirection.UNKNOWN:
            trajectory_data.movement_direction = detected_direction
    
    def get_trajectory_direction(self, global_id: int) -> MovementDirection:
        if global_id not in self.global_trajectories:
            return MovementDirection.UNKNOWN
        return self.global_trajectories[global_id].movement_direction
    
    def is_direction_stable(self, global_id: int) -> bool:
        if global_id not in self.global_trajectories:
            return False
        return self.global_trajectories[global_id].direction_locked
    
    def cleanup_stale_trajectories(self, current_time: float, prediction_timeout: float = 10.0):
        effective_timeout = min(self.trajectory_timeout, prediction_timeout)

        stale_local_keys = [
            key for key, data in self.local_vehicle_info.items()
            if current_time - data['timestamp'] > effective_timeout
        ]
        
        for key in stale_local_keys:
            del self.local_vehicle_info[key]
        
        stale_global_ids = [
            global_id for global_id, data in self.global_trajectories.items()
            if current_time - data.last_update > effective_timeout
        ]
        
        for global_id in stale_global_ids:
            logger.debug(f"清理过期轨迹: 车辆 {global_id}")
            self.global_trajectories.pop(global_id, None)
            self.global_speeds.pop(global_id, None)
            self.global_lanes.pop(global_id, None)
    
    def get_vehicle_positions_for_speed_calculation(self, latest_update_time: float, 
                                                prediction_manager: VehiclePredictionManager) -> Dict:
        vehicle_positions = {}
        
        for global_id, data in self.global_trajectories.items():
            if len(data.trajectory) > 0:
                last_position = data.trajectory[-1]
                state = prediction_manager.get_vehicle_state(global_id)
                
                if state == VehicleState.LOST:
                    camera_id = f'{state.value}_prediction'
                else:
                    params = prediction_manager.state_manager.motion_params.get(global_id)
                    camera_id = params.last_camera if params else 'unknown'
                
                vehicle_positions[global_id] = {
                    'position': last_position,
                    'timestamp': data.last_update,
                    'camera_id': camera_id,
                    'state': state.value,
                    'direction': data.movement_direction.value
                }
        
        return vehicle_positions


class TunnelRenderer:
    
    def __init__(self, map_width: int, map_height: int):
        self.map_width = map_width
        self.map_height = map_height
        self.label_area_height = 50
        self.total_height = map_height + self.label_area_height
        
        self.base_map = self._create_base_map()
    
    def _create_base_map(self) -> np.ndarray:
        map_image = np.ones((self.total_height, self.map_width, 3), dtype=np.uint8) * 255
        
        self._draw_lane_lines(map_image)
        
        self._draw_camera_regions(map_image)
        
        self._draw_blind_zones(map_image)
        
        self._draw_boundaries_and_coordinates(map_image)
        
        return map_image
    
    def _draw_lane_lines(self, map_image: np.ndarray):
        for i, lane in enumerate(config.MAP_LANE_DEFINITIONS):
            if i < len(config.MAP_LANE_DEFINITIONS) - 1:
                lane_divider = lane["y_range"][1]
                DrawingUtils.draw_dashed_line(
                    map_image, (0, lane_divider), (self.map_width, lane_divider),
                    (120, 120, 120), 2, 20, 10
                )
            
            cv2.putText(map_image, lane["name"], 
                       (15, (lane["y_range"][0] + lane["y_range"][1]) // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    
    def _draw_camera_regions(self, map_image: np.ndarray):
        colors = {
            'cam1': (255, 220, 220),
            'cam2': (220, 255, 220),
            'cam3': (220, 220, 255),
            'cam4': (255, 255, 200),
            'cam5': (200, 255, 255),
            'cam6': (255, 200, 255)
        }
        
        for cam_id, cam_config in config.CAMERA_CONFIG.items():
            region = cam_config['calibration']['map_region']
            x1, y1, x2, y2 = [max(0, min(coord, self.map_width if i % 2 == 0 else self.map_height)) 
                              for i, coord in enumerate(region)]
            
            overlay = map_image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors.get(cam_id, (240, 240, 240)), -1)
            cv2.addWeighted(overlay, 0.3, map_image, 0.7, 0, map_image)
    
    def _draw_blind_zones(self, map_image: np.ndarray):
        for (cam1, cam2), zone_info in config.BLIND_ZONES.items():
            start_x, end_x = zone_info['start_x'], zone_info['end_x']
            
            overlay = map_image.copy()
            cv2.rectangle(overlay, (start_x, 0), (end_x, self.map_height), (200, 200, 200), -1)
            cv2.addWeighted(overlay, 0.4, map_image, 0.6, 0, map_image)
            
            center_x = (start_x + end_x) // 2
            cv2.putText(map_image, "BLIND ZONE", (center_x - 35, self.map_height + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, lineType=cv2.LINE_AA)
    
    def _draw_boundaries_and_coordinates(self, map_image: np.ndarray):
        cv2.line(map_image, (0, 5), (self.map_width, 5), (100, 100, 100), 2)
        cv2.line(map_image, (0, self.map_height), (self.map_width, self.map_height), (100, 100, 100), 2)
        
        tunnel_length = config.TUNNEL_REAL_DIMENSIONS['length']
        for real_distance in range(50, int(tunnel_length) + 1, 50):
            pixel_x, _ = config.real_to_pixel_coordinates(real_distance, 0)
            if 0 <= pixel_x < self.map_width:
                cv2.putText(map_image, f"{real_distance}m", (pixel_x, self.map_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, lineType=cv2.LINE_AA)
        
        self._draw_camera_labels(map_image)
    
    def _draw_camera_labels(self, map_image: np.ndarray):
        for cam_id, cam_config in config.CAMERA_CONFIG.items():
            try:
                region = cam_config['calibration']['map_region']
                x1, y1, x2, y2 = [max(0, min(coord, self.map_width if i % 2 == 0 else self.map_height)) 
                                  for i, coord in enumerate(region)]
                
                center_x = (x1 + x2) // 2
                
                if 0 <= center_x < self.map_width:
                    label_y = self.map_height + 35
                    
                    text_size, _ = cv2.getTextSize(cam_id.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_width = text_size[0]
                    
                    text_x = max(2, min(center_x - text_width // 2, self.map_width - text_width - 2))
                    
                    cv2.rectangle(map_image, 
                                (text_x - 2, label_y - 12), 
                                (text_x + text_width + 2, label_y + 2),
                                (240, 240, 240), -1)
                    cv2.rectangle(map_image, 
                                (text_x - 2, label_y - 12), 
                                (text_x + text_width + 2, label_y + 2),
                                (150, 150, 150), 1)
                    
                    cv2.putText(map_image, cam_id.upper(), (text_x, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                               
            except Exception as e:
                logger.warning(f"绘制摄像头 {cam_id} 标签时出错: {e}")
                continue
    
    def render_trajectories(self, map_image: np.ndarray, trajectory_manager: VehicleTrajectoryManager,
                          prediction_manager: VehiclePredictionManager, calculated_speeds: Dict,
                          current_time: float) -> np.ndarray:
        result_map = map_image.copy()

        for global_id, data in trajectory_manager.global_trajectories.items():
            if len(data.trajectory) < 1:
                continue

            self._draw_single_trajectory(result_map, global_id, data, prediction_manager,
                                       calculated_speeds, current_time)

        return result_map

    def render_trajectories_from_snapshot(self, map_image: np.ndarray,
                                         global_trajectories: Dict[int, TrajectoryData],
                                         prediction_manager: VehiclePredictionManager,
                                         calculated_speeds: Dict,
                                         current_time: float) -> np.ndarray:
        result_map = map_image.copy()

        for global_id, data in global_trajectories.items():
            if len(data.trajectory) < 1:
                continue

            self._draw_single_trajectory(result_map, global_id, data, prediction_manager,
                                       calculated_speeds, current_time)

        return result_map
    
    def _draw_single_trajectory(self, map_image: np.ndarray, global_id: int, 
                            trajectory_data: TrajectoryData, prediction_manager: VehiclePredictionManager,
                            calculated_speeds: Dict, current_time: float):
        trajectory = trajectory_data.trajectory
        color = trajectory_data.color
        state = prediction_manager.get_vehicle_state(global_id)
        movement_direction = trajectory_data.movement_direction
        
        if len(trajectory) >= 2:
            points = np.array(list(trajectory), dtype=np.int32)
            self._draw_trajectory_lines(map_image, points, color, state, movement_direction)
        
        current_pos = None
        
        if state == VehicleState.LOST:
            if len(trajectory) > 0:
                current_pos = list(trajectory)[-1]
        else:
            if trajectory_data.last_real_position is not None:
                current_pos = trajectory_data.last_real_position
            elif len(trajectory) > 0:
                current_pos = list(trajectory)[-1]
        
        if current_pos is None:
            logger.debug(f"车辆 {global_id} 没有有效位置，跳过渲染")
            return
        
        if not (0 <= current_pos[0] < self.map_width and 0 <= current_pos[1] < self.map_height):
            logger.debug(f"车辆 {global_id} 位置超出范围 {current_pos}，跳过渲染")
            return
        
        self._draw_vehicle_marker(map_image, current_pos, color, state)
        self._draw_vehicle_label(map_image, current_pos, global_id, state, 
                            calculated_speeds, prediction_manager, current_time, movement_direction)
    
    def _draw_trajectory_lines(self, map_image: np.ndarray, points: np.ndarray,
                             color: Tuple[int, int, int], state: VehicleState, 
                             movement_direction: MovementDirection):
        for i in range(1, len(points)):
            pt1, pt2 = tuple(points[i-1]), tuple(points[i])
            
            if (0 <= pt1[0] < self.map_width and 0 <= pt1[1] < self.map_height and
                0 <= pt2[0] < self.map_width and 0 <= pt2[1] < self.map_height):
                
                cv2.line(map_image, pt1, pt2, color, 2)
    
    def _draw_vehicle_marker(self, map_image: np.ndarray, position: Tuple[int, int],
                        color: Tuple[int, int, int], state: VehicleState):
        if state == VehicleState.LOST:
            points = np.array([
                [position[0] - 5, position[1] - 6],
                [position[0] + 5, position[1] - 6],
                [position[0] + 5, position[1] + 6],
                [position[0] - 5, position[1] + 6]
            ], np.int32)
            cv2.fillPoly(map_image, [points], color)
            cv2.polylines(map_image, [points], True, (255, 255, 255), 1)
        else:
            cv2.circle(map_image, position, 6, color, -1)
            cv2.circle(map_image, position, 6, (255, 255, 255), 1)
    
    def _draw_vehicle_label(self, map_image: np.ndarray, position: Tuple[int, int],
                          global_id: int, state: VehicleState, calculated_speeds: Dict,
                          prediction_manager: VehiclePredictionManager, current_time: float,
                          movement_direction: MovementDirection):
        speed_text = ""
        if global_id in calculated_speeds:
            speed = calculated_speeds[global_id]
            speed_text = f" {speed:.0f} km/h"
        
        direction_text = ""
        if movement_direction != MovementDirection.UNKNOWN:
            direction_symbols = {
                MovementDirection.LEFT: "←",
                MovementDirection.RIGHT: "→",
                MovementDirection.STATIONARY: "●"
            }
            direction_text = f" {direction_symbols.get(movement_direction, '')}"
        
        label_text = f"ID:{global_id}{speed_text}"
        
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_width, text_height = text_size
        
        text_x = max(2, min(position[0] - text_width // 2, self.map_width - text_width - 2))
        text_y = min(position[1] + 20, self.map_height - 5)
        
        bg_color = (255, 255, 255)
        cv2.rectangle(map_image, 
                     (text_x - 2, text_y - text_height - 2),
                     (text_x + text_width + 2, text_y + 2),
                     bg_color, -1)
        cv2.rectangle(map_image, 
                     (text_x - 2, text_y - text_height - 2),
                     (text_x + text_width + 2, text_y + 2),
                     (100, 100, 100), 1)
        
        text_color = (0, 0, 0)
        cv2.putText(map_image, label_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, lineType=cv2.LINE_AA)
    
    def create_info_panel(self, trajectory_manager: VehicleTrajectoryManager,
                         prediction_manager: VehiclePredictionManager, current_time: float,
                         fps: float, camera_last_update: Dict,
                         predictors: Dict) -> np.ndarray:
        info_panel_height = 150
        info_panel = np.ones((info_panel_height, self.map_width, 3), dtype=np.uint8) * 255

        total_vehicles = len(trajectory_manager.global_trajectories)
        state_counts = defaultdict(int)
        for global_id in trajectory_manager.global_trajectories.keys():
            state = prediction_manager.get_vehicle_state(global_id)
            state_counts[state] += 1

        cv2.line(info_panel, (0, 0), (self.map_width, 0), (200, 200, 200), 2)
        col_widths = [int(self.map_width * 0.25), int(self.map_width * 0.5), int(self.map_width * 0.75)]
        for col_x in col_widths:
            cv2.line(info_panel, (col_x, 0), (col_x, info_panel_height), (220, 220, 220), 1)

        self._draw_vehicle_statistics(info_panel, 10, total_vehicles, state_counts)

        self._draw_system_info(info_panel, col_widths[0] + 10, fps, len(predictors))

        self._draw_camera_info(info_panel, col_widths[1] + 10, current_time, camera_last_update)

        self._draw_tunnel_info(info_panel, col_widths[2] + 10)

        return info_panel

    def create_info_panel_from_snapshot(self, global_trajectories: Dict[int, TrajectoryData],
                                       prediction_manager: VehiclePredictionManager, current_time: float,
                                       fps: float, camera_last_update: Dict,
                                       predictors: Dict) -> np.ndarray:
        info_panel_height = 150
        info_panel = np.ones((info_panel_height, self.map_width, 3), dtype=np.uint8) * 255

        total_vehicles = len(global_trajectories)
        state_counts = defaultdict(int)
        for global_id in global_trajectories.keys():
            if prediction_manager:
                state = prediction_manager.get_vehicle_state(global_id)
                state_counts[state] += 1

        cv2.line(info_panel, (0, 0), (self.map_width, 0), (200, 200, 200), 2)
        col_widths = [int(self.map_width * 0.25), int(self.map_width * 0.5), int(self.map_width * 0.75)]
        for col_x in col_widths:
            cv2.line(info_panel, (col_x, 0), (col_x, info_panel_height), (220, 220, 220), 1)

        self._draw_vehicle_statistics(info_panel, 10, total_vehicles, state_counts)

        self._draw_system_info(info_panel, col_widths[0] + 10, fps, len(predictors))

        self._draw_camera_info(info_panel, col_widths[1] + 10, current_time, camera_last_update)

        self._draw_tunnel_info(info_panel, col_widths[2] + 10)

        return info_panel
    
    def _draw_vehicle_statistics(self, info_panel: np.ndarray, x: int, total_vehicles: int, state_counts: Dict):
        cv2.putText(info_panel, f"Total Vehicles: {total_vehicles}", 
                (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(info_panel, f"Detected: {state_counts[VehicleState.DETECTED]}", 
                (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(info_panel, f"Lost/Predicted: {state_counts[VehicleState.LOST]}", 
                (x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1, lineType=cv2.LINE_AA)
    
    def _draw_system_info(self, info_panel: np.ndarray, x: int, fps: float, active_predictors: int):
        cv2.putText(info_panel, f"FPS: {fps:.1f}", 
                   (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(info_panel, f"Active Predictors: {active_predictors}", 
                   (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(info_panel, f"Map Resolution:", 
                   (x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(info_panel, f"{config.COORDINATE_CALIBRATION['meters_per_pixel_x']:.3f} m/px", 
                   (x, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    
    def _draw_camera_info(self, info_panel: np.ndarray, x: int, current_time: float, camera_last_update: Dict):
        y_offset = 25
        for cam_id in sorted(config.CAMERA_CONFIG.keys()):
            if cam_id in camera_last_update:
                time_since_last = current_time - camera_last_update[cam_id]
                cv2.putText(info_panel, f"{cam_id}: {time_since_last:.0f}s ahead", 
                           (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            else:
                cv2.putText(info_panel, f"{cam_id}: No data", 
                           (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, lineType=cv2.LINE_AA)
            y_offset += 20
    
    def _draw_tunnel_info(self, info_panel: np.ndarray, x: int):
        cv2.putText(info_panel, f"Tunnel: {config.TUNNEL_REAL_DIMENSIONS['length']:.0f}m", 
                   (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(info_panel, f"Map: {self.map_width}x{self.map_height}", 
                   (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)


class TunnelMap:
    
    def __init__(self, map_width: Optional[int] = None, map_height: Optional[int] = None, camera_calibrations: Optional[Dict] = None):
        self.map_width = map_width or config.MAP_WIDTH
        self.map_height = map_height or config.MAP_HEIGHT
        self.camera_calibrations = camera_calibrations or {}
        
        self.renderer = TunnelRenderer(self.map_width, self.map_height)
        self.speed_calculator = VehicleSpeedCalculator(smooth_factor=0.7)
        
        self.prediction_manager = None
        
        self.lock = threading.RLock()
        self.latest_update_time = 0
        self.camera_last_update: Dict[str, float] = {}
        
        windows = config.TRACKING_STRATEGY['time_windows']
        ghost_scoring = config.TRACKING_STRATEGY.get('ghost_scoring', {})
        self.max_trajectory_length = 20
        self.trajectory_timeout = float(windows['trajectory_timeout_sec'])
        self.prediction_timeout = float(windows['prediction_timeout_sec'])
        self.ghost_no_detection_threshold = float(windows['ghost_no_detection_sec'])
        self.blind_zone_protection_enabled = bool(windows['blind_zone_protection_enabled'])

        self.ghost_scoring_enabled = bool(ghost_scoring.get('enabled', True))
        self.ghost_enter_threshold = float(ghost_scoring.get('enter_score_threshold', 0.60))
        self.ghost_exit_threshold = float(ghost_scoring.get('exit_score_threshold', 0.45))
        self.ghost_delete_threshold = float(ghost_scoring.get('delete_score_threshold', 0.75))
        self.ghost_lane_tolerance = float(ghost_scoring.get('lane_tolerance_pixels', 12.0))
        self.ghost_inactive_floor_sec = float(ghost_scoring.get('inactive_floor_sec', 1.0))
        self.ghost_score_weights: Dict[str, float] = {
            'time_window': float(ghost_scoring.get('weights', {}).get('time_window', 0.35)),
            'tracking_inactive': float(ghost_scoring.get('weights', {}).get('tracking_inactive', 0.25)),
            'predictor_health': float(ghost_scoring.get('weights', {}).get('predictor_health', 0.15)),
            'lane_consistency': float(ghost_scoring.get('weights', {}).get('lane_consistency', 0.10)),
            'direction_consistency': float(ghost_scoring.get('weights', {}).get('direction_consistency', 0.10)),
            'camera_region': float(ghost_scoring.get('weights', {}).get('camera_region', 0.05)),
        }

        self.trajectory_manager = VehicleTrajectoryManager(
            max_trajectory_length=self.max_trajectory_length,
            trajectory_timeout=self.trajectory_timeout
        )
        
        self.vehicle_speed_cache: Dict[int, float] = {}
        self.detection_region_boundaries: Dict[str, Tuple[int, int]] = {}
        
        self.vehicle_classes: Dict[int, int] = {}

        self.vehicle_lane_cache: Dict[int, Dict[str, Any]] = {}
        
        self.lane_classifiers = {}
        for cam_id in config.CAMERA_CONFIG.keys():
            if cam_id in self.camera_calibrations:
                self.lane_classifiers[cam_id] = LaneClassifier(cam_id)
            else:
                try:
                    self.lane_classifiers[cam_id] = LaneClassifier(cam_id)
                except Exception as e:
                    logger.warning(f"无法为摄像头 {cam_id} 创建车道分类器: {e}")

        logger.info(f"车道分类器: {list(self.lane_classifiers.keys())}")
        
        self.camera_boundaries = self._calculate_camera_boundaries()
        
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps = 0
        
        self.lane_mapping = {}
        for cam_id in config.CAMERA_CONFIG.keys():
            self.lane_mapping[cam_id] = {0: 0, 1: 1, 2: 2}

        logger.info(f"支持的摄像头: {list(self.lane_mapping.keys())}")
        
        self._initialize_detection_boundaries()
        
        self.last_ghost_check_time = 0
        self.ghost_check_interval = float(windows['ghost_check_interval_sec'])
        self.ghost_suspect_window = float(windows['ghost_suspect_window_sec'])
        self.ghost_quarantine_window = float(windows['ghost_quarantine_window_sec'])
        self.ghost_event_log: Deque[Dict[str, Any]] = deque(maxlen=int(windows['ghost_event_log_limit']))
        self.ghost_event_seq = 0
        self.ghost_lifecycle_state: Dict[int, Dict[str, Any]] = {}
        
        logger.info(f"隧道地图初始化完成 - 等待外部设置预测管理器，启用detection_region速度计算")
    
    def set_prediction_manager(self, prediction_manager: VehiclePredictionManager):
        self.prediction_manager = prediction_manager
        logger.info("隧道地图已设置预测管理器")
    
    @property
    def state_manager(self):
        if self.prediction_manager is None:
            return None
        return self.prediction_manager.state_manager
    
    @property
    def blind_zone_predictors(self):
        if self.prediction_manager is None:
            return {}
        return self.prediction_manager.predictors
    
    def start_lost_vehicle_prediction(self, global_id: int, last_position: Tuple[int, int],
                                    speed: Tuple[float, float], direction: float, 
                                    lane_y: float, start_time: float, last_camera: str) -> bool:
        if self.prediction_manager is None:
            return False
        return self.prediction_manager.start_prediction(
            global_id, last_position, speed, direction, lane_y, start_time, last_camera
        )
    
    def _calculate_camera_boundaries(self) -> Dict:
        boundaries = {}
        for cam_id, cam_config in config.CAMERA_CONFIG.items():
            region = cam_config['calibration']['map_region']
            x_min, y_min, x_max, y_max = region
            boundaries[cam_id] = {
                'left': x_min, 'right': x_max,
                'top': y_min, 'bottom': y_max
            }
        return boundaries
    
    def _initialize_detection_boundaries(self):
        for cam_id in config.CAMERA_CONFIG.keys():
            boundary_result = config.get_camera_detection_boundaries(cam_id)
            if boundary_result[0] is not None and boundary_result[1] is not None:
                self.detection_region_boundaries[cam_id] = (boundary_result[0], boundary_result[1])
                logger.info(f"摄像头 {cam_id} 检测区域边界: {boundary_result[0]} - {boundary_result[1]} 地图像素")
            else:
                logger.warning(f"摄像头 {cam_id} 无法获取检测区域边界，将不计算速度")
    
    def _is_in_detection_region(self, camera_id: str, position: Tuple[int, int]) -> bool:
        if camera_id not in self.detection_region_boundaries:
            return False
        
        min_x, max_x = self.detection_region_boundaries[camera_id]
        x, y = position
        return min_x <= x <= max_x
    
    def _calculate_speed_in_detection_region(self, global_id: int, camera_id: str, 
                                           position: Tuple[int, int], current_time: float):
        if not self._is_in_detection_region(camera_id, position):
            return
        
        vehicle_positions = {
            global_id: {
                'position': position,
                'timestamp': current_time,
                'camera_id': camera_id,
                'state': 'detected',
                'direction': (self.prediction_manager.get_movement_direction(global_id).value 
                         if self.prediction_manager else MovementDirection.UNKNOWN.value)
            }
        }
        
        calculated_speeds = self.speed_calculator.update_vehicle_positions_from_map(vehicle_positions)
        
        if global_id in calculated_speeds:
            new_speed = calculated_speeds[global_id]
            if 0 < new_speed < 120:
                self.vehicle_speed_cache[global_id] = new_speed
                logger.info(f"车辆 {global_id} 在摄像头 {camera_id} 检测区域内计算速度: {new_speed:.1f} km/h")
    
    def bbox_to_map_position(self, camera_id: str, bbox: List) -> Optional[Tuple[int, int]]:
        try:
            bottom_center = [int((bbox[0] + bbox[2]) / 2), int(bbox[3])]
            lane_index, _ = self._classify_lane(camera_id, bbox)
            return self._map_coordinates(camera_id, bottom_center, lane_index)
        except Exception:
            return None

    def get_vehicle_lane_for_bbox(self, global_id: int, camera_id: str, bbox: List,
                                  current_time: Optional[float] = None) -> Tuple[int, str]:
        return self._classify_lane(camera_id, bbox, global_id, current_time)

    def update_vehicle_position(self, global_id: int, camera_id: str, bbox: List, 
                               current_time: float, class_id: Optional[int] = None) -> bool:
        with self.lock:
            self.latest_update_time = current_time
            self.camera_last_update[camera_id] = current_time
            
            if class_id is not None:
                self.vehicle_classes[global_id] = class_id
            
            if self.prediction_manager is None:
                current_state = VehicleState.DETECTED
                was_in_prediction = False
            else:
                current_state = self.prediction_manager.get_vehicle_state(global_id)
                was_in_prediction = current_state == VehicleState.LOST
            
            try:
                map_position = self._process_vehicle_position(global_id, camera_id, bbox, current_time, was_in_prediction)
                if map_position is not None:
                    self._clear_ghost_lifecycle(global_id, current_time, "real_detection")

                    if was_in_prediction and self.prediction_manager is not None:
                        logger.info(f"车辆 {global_id} 从{current_state.value}状态重新被摄像头 {camera_id} 检测到")
                        self.prediction_manager.handle_vehicle_detected(
                            global_id, camera_id, map_position, current_time, True
                        )

                    self._detect_and_cleanup_ghost_vehicles(camera_id, current_time)
                    return True

                return False
                    
            except Exception as e:
                logger.error(f"更新车辆位置时出错: {e}")
                return False
        
        return False
    
    def _process_vehicle_position(self, global_id: int, camera_id: str, bbox: List, 
                                current_time: float, was_in_prediction: bool) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = bbox
        bottom_center = [int((x1 + x2) / 2), int(y2)]
        
        lane_index, lane_name = self._classify_lane(camera_id, bbox, global_id, current_time)
        
        map_position = self._map_coordinates(camera_id, bottom_center, lane_index)
        if map_position is None:
            return None
        
        map_position = self._ensure_continuous_movement(global_id, map_position)
        
        x, y = map_position
        
        if not was_in_prediction:
            self._update_motion_params(global_id, (x, y), current_time, y)
        
        self._update_vehicle_state(global_id, was_in_prediction)
        
        self._update_trajectories(global_id, camera_id, (x, y), current_time, bbox, lane_index, lane_name)
        
        return (x, y)
    
    def _classify_lane(self, camera_id: str, bbox: List, global_id: Optional[int] = None,
                       current_time: Optional[float] = None) -> Tuple[int, str]:
        if camera_id not in self.lane_classifiers:
            cached_lane = self._get_cached_vehicle_lane(global_id)
            if cached_lane is not None:
                return cached_lane
            return 0, "default_lane"

        lane_index, lane_name = self.lane_classifiers[camera_id].determine_lane(bbox)
        if lane_index is None or lane_index < 0:
            cached_lane = self._get_cached_vehicle_lane(global_id)
            if cached_lane is not None:
                logger.debug(f"车辆 {global_id} 越出车道边界，沿用历史车道: {cached_lane[1]}")
                return cached_lane
            return 0, "out_of_lane_default"

        self._update_vehicle_lane_cache(global_id, camera_id, lane_index, lane_name, current_time)
        return lane_index, lane_name

    def _get_cached_vehicle_lane(self, global_id: Optional[int]) -> Optional[Tuple[int, str]]:
        if global_id is None:
            return None

        lane_info = self.vehicle_lane_cache.get(global_id)
        if lane_info is None:
            return None

        return int(lane_info['lane_index']), str(lane_info['lane_name'])

    def _update_vehicle_lane_cache(self, global_id: Optional[int], camera_id: str,
                                   lane_index: int, lane_name: str,
                                   current_time: Optional[float]) -> None:
        if global_id is None or lane_index < 0:
            return

        self.vehicle_lane_cache[global_id] = {
            'lane_index': int(lane_index),
            'lane_name': lane_name,
            'camera_id': camera_id,
            'updated_at': current_time if current_time is not None else self.latest_update_time,
        }
    
    def _map_coordinates(self, camera_id: str, bottom_center: List, lane_index: int) -> Optional[Tuple[int, int]]:
        map_lane_index = self.lane_mapping[camera_id].get(lane_index, 0)
        if map_lane_index >= len(config.MAP_LANE_DEFINITIONS):
            map_lane_index = 0
        
        map_lane = config.MAP_LANE_DEFINITIONS[map_lane_index]
        
        if camera_id in self.camera_calibrations:
            try:
                map_position = self.camera_calibrations[camera_id].map_to_ground(bottom_center)
                rel_x = map_position[0]
            except Exception as e:
                logger.warning(f"坐标映射错误: {e}")
                return None
        else:
            logger.error(f"未找到摄像头 {camera_id} 的标定信息")
            return None
        
        cam_region = config.CAMERA_CONFIG[camera_id]['calibration']['map_region']
        x_min, _, x_max, _ = cam_region
        
        boundary_extension = 50
        extended_x_min = max(0, x_min - boundary_extension)
        extended_x_max = min(config.MAP_WIDTH, x_max + boundary_extension)
        
        lane_y_min, lane_y_max = map_lane["y_range"]
        x = np.clip(rel_x, extended_x_min, extended_x_max)
        y = (lane_y_min + lane_y_max) / 2
        
        return int(x), int(y)
    
    def _update_motion_params(self, global_id: int, current_pos: Tuple[int, int], 
                            current_time: float, lane_y: float):
        if self.prediction_manager is None:
            return
        if global_id in self.trajectory_manager.global_trajectories:
            data = self.trajectory_manager.global_trajectories[global_id]
            if len(data.trajectory) > 0:
                last_pos = data.trajectory[-1]
                time_diff = current_time - data.last_update
                if time_diff > 0:
                    self.prediction_manager.update_motion_params(
                        global_id, current_pos, last_pos, time_diff, lane_y
                    )
    
    def _update_vehicle_state(self, global_id: int, was_in_prediction: bool):
        if self.prediction_manager is None:
            return
        if was_in_prediction:
            self.prediction_manager.set_vehicle_state(global_id, VehicleState.DETECTED)
            logger.info(f"车辆 {global_id} 从预测状态恢复为检测状态")
        else:
            current_state = self.prediction_manager.get_vehicle_state(global_id)
            if current_state != VehicleState.DETECTED:
                self.prediction_manager.set_vehicle_state(global_id, VehicleState.DETECTED)
    
    def _update_trajectories(self, global_id: int, camera_id: str, position: Tuple[int, int],
                           current_time: float, bbox: List, lane_index: int, lane_name: str):
        movement_direction = (self.prediction_manager.get_movement_direction(global_id) 
                             if self.prediction_manager else MovementDirection.UNKNOWN)
        
        self._calculate_speed_in_detection_region(global_id, camera_id, position, current_time)
        
        vehicle_info = {
            'camera_id': camera_id,
            'global_id': global_id,
            'bbox': bbox,
            'position': position,
            'lane_index': lane_index,
            'lane_name': lane_name,
            'timestamp': current_time,
            'movement_direction': movement_direction.value
        }
        
        self.trajectory_manager.update_local_trajectory(
            camera_id, global_id, position, current_time, vehicle_info
        )
        
        self.trajectory_manager.update_global_trajectory(
            global_id, position, current_time, camera_id, position[1], movement_direction
        )
    
    def render(self, current_time: Optional[float] = None) -> np.ndarray:
        with self.lock:
            if current_time is None:
                current_time = self.latest_update_time or time.time()

            if self.prediction_manager is None:
                logger.warning("预测管理器未设置，返回基础地图")
                return cv2.vconcat([self.renderer.base_map.copy(),
                                  np.ones((150, self.map_width, 3), dtype=np.uint8) * 255])

            self._update_fps()

            self._update_predictions(current_time)

            self._cleanup_data(current_time)

            if not self.vehicle_speed_cache:
                vehicle_positions = self.trajectory_manager.get_vehicle_positions_for_speed_calculation(
                    self.latest_update_time, self.prediction_manager
                )
                initial_speeds = self.speed_calculator.update_vehicle_positions_from_map(vehicle_positions)
                for global_id, speed in initial_speeds.items():
                    if global_id not in self.vehicle_speed_cache and 0 < speed < 200:
                        self.vehicle_speed_cache[global_id] = speed

            quarantine_ids = {
                global_id for global_id, state in self.ghost_lifecycle_state.items()
                if state.get('state') == 'quarantine'
            }

            if quarantine_ids:
                global_trajectories_snapshot = {
                    global_id: data for global_id, data in self.trajectory_manager.global_trajectories.items()
                    if global_id not in quarantine_ids
                }
            else:
                global_trajectories_snapshot = dict(self.trajectory_manager.global_trajectories)

            prediction_manager = self.prediction_manager
            speed_cache = {
                global_id: speed for global_id, speed in self.vehicle_speed_cache.items()
                if global_id not in quarantine_ids
            }
            fps = self.fps
            camera_last_update = dict(self.camera_last_update)
            predictors_snapshot = {
                global_id: predictor for global_id, predictor in prediction_manager.predictors.items()
                if global_id not in quarantine_ids
            } if prediction_manager else {}

        map_with_trajectories = self.renderer.render_trajectories_from_snapshot(
            self.renderer.base_map.copy(), global_trajectories_snapshot,
            prediction_manager, speed_cache, current_time
        )

        info_panel = self.renderer.create_info_panel_from_snapshot(
            global_trajectories_snapshot, prediction_manager, current_time,
            fps, camera_last_update, predictors_snapshot
        )

        return cv2.vconcat([map_with_trajectories, info_panel])
    
    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.last_fps_update
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def _update_predictions(self, current_time: float):
        if self.prediction_manager is None:
            return
        
        predicted_positions = self.prediction_manager.update_predictions(current_time)
        
        for global_id, predicted_position in predicted_positions.items():
            if global_id in self.trajectory_manager.global_trajectories:
                data = self.trajectory_manager.global_trajectories[global_id]
                data.trajectory.append(predicted_position)
                data.last_update = current_time
                
                predictor_info = self.prediction_manager.get_predictor_info(global_id)
                if predictor_info:
                    predicted_speed = predictor_info['speed_kmh']
                    self.trajectory_manager.global_speeds[global_id] = predicted_speed
                    
                    if global_id not in self.vehicle_speed_cache:
                        self.vehicle_speed_cache[global_id] = predicted_speed
                        logger.debug(f"为没有速度记录的预测车辆 {global_id} 设置速度: {predicted_speed:.1f} km/h")
    
    def _cleanup_data(self, current_time: float):
        self.trajectory_manager.cleanup_stale_trajectories(current_time, self.prediction_timeout)
        if self.prediction_manager is not None:
            self.prediction_manager.cleanup_expired_predictions(current_time)
        
        if current_time - self.last_ghost_check_time >= self.ghost_check_interval:
            self.last_ghost_check_time = current_time
            if hasattr(self, 'id_manager') and self.prediction_manager is not None:
                logger.debug("执行定期幽灵车辆检测")
                self.trigger_ghost_vehicle_detection(current_time=current_time)
        
        active_vehicle_ids = set(self.trajectory_manager.global_trajectories.keys())
        cached_vehicle_ids = set(self.vehicle_speed_cache.keys())
        cached_lane_ids = set(self.vehicle_lane_cache.keys())

        stale_lifecycle_ids = set(self.ghost_lifecycle_state.keys()) - active_vehicle_ids
        for global_id in stale_lifecycle_ids:
            self._clear_ghost_lifecycle(global_id, current_time, "vehicle_not_active")

        for global_id in cached_vehicle_ids - active_vehicle_ids:
            logger.debug(f"清理车辆 {global_id} 的速度缓存")
            self.vehicle_speed_cache.pop(global_id, None)
            self.vehicle_classes.pop(global_id, None)

        for global_id in cached_lane_ids - active_vehicle_ids:
            logger.debug(f"清理车辆 {global_id} 的车道缓存")
            self.vehicle_lane_cache.pop(global_id, None)
    
    def _ensure_continuous_movement(self, global_id: int, new_position: Tuple[int, int]) -> Tuple[int, int]:
        x, y = new_position
        
        if x < 0:
            x = 0
        elif x >= self.map_width:
            x = self.map_width - 1
        
        if y < 0:
            y = 0
        elif y >= self.map_height:
            y = self.map_height - 1
        
        clamped_position = (int(x), int(y))
        
        if clamped_position != new_position:
            logger.debug(f"车辆 {global_id} 位置被约束到地图边界内: {new_position} -> {clamped_position}")
        
        return clamped_position

    def _is_position_in_blind_zone(self, position: Tuple[int, int]) -> bool:
        x, _ = position
        for zone_info in config.BLIND_ZONES.values():
            start_x = zone_info.get('start_x')
            end_x = zone_info.get('end_x')
            if start_x is None or end_x is None:
                continue
            if start_x <= x <= end_x:
                return True
        return False

    def _is_prediction_healthy(self, global_id: int, current_time: float) -> bool:
        if self.prediction_manager is None:
            return False

        predictor = self.prediction_manager.predictors.get(global_id)
        if predictor is None or not predictor.is_active:
            return False

        max_lifetime = getattr(predictor, 'max_prediction_time', self.prediction_timeout)
        return (current_time - predictor.start_time) <= max_lifetime

    def _should_protect_blind_zone_vehicle(self, global_id: int,
                                         position: Tuple[int, int],
                                         current_time: float) -> bool:
        if not self.blind_zone_protection_enabled:
            return False
        return self._is_position_in_blind_zone(position) and self._is_prediction_healthy(global_id, current_time)

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _get_camera_index(self, camera_id: str) -> Optional[int]:
        camera_order = list(config.CAMERA_CONFIG.keys())
        try:
            return camera_order.index(camera_id)
        except ValueError:
            return None

    def _compute_lane_inconsistency(self, global_id: int, position: Tuple[int, int]) -> float:
        if self.prediction_manager is None:
            return 0.5

        predictor = self.prediction_manager.predictors.get(global_id)
        if predictor is not None:
            reference_y = float(getattr(predictor, 'lane_y', position[1]))
        else:
            info = self.prediction_manager.vehicle_tracking_info.get(global_id)
            if info is None:
                return 0.5
            reference_y = float(info.last_position[1])

        lane_diff = abs(float(position[1]) - reference_y)
        if lane_diff <= self.ghost_lane_tolerance:
            return 0.0

        return self._clamp_01((lane_diff - self.ghost_lane_tolerance) / max(self.ghost_lane_tolerance * 2.0, 1.0))

    def _compute_direction_inconsistency(self, global_id: int, trigger_camera_id: str) -> float:
        if self.prediction_manager is None:
            return 0.5

        trajectory_data = self.trajectory_manager.global_trajectories.get(global_id)
        if trajectory_data is None:
            return 0.5

        trajectory_direction = trajectory_data.movement_direction
        predictor = self.prediction_manager.predictors.get(global_id)
        predictor_direction = predictor.movement_direction if predictor is not None else MovementDirection.UNKNOWN

        if trajectory_direction == MovementDirection.UNKNOWN:
            direction_component = 0.5
        elif predictor_direction == MovementDirection.UNKNOWN:
            direction_component = 0.4
        else:
            direction_component = 0.0 if trajectory_direction == predictor_direction else 1.0

        camera_component = 0.5
        info = self.prediction_manager.vehicle_tracking_info.get(global_id)
        if info is not None:
            src_index = self._get_camera_index(info.last_camera)
            dst_index = self._get_camera_index(trigger_camera_id)
            if src_index is not None and dst_index is not None:
                delta = dst_index - src_index
                if trajectory_direction == MovementDirection.RIGHT:
                    camera_component = 0.0 if delta >= 0 else 1.0
                elif trajectory_direction == MovementDirection.LEFT:
                    camera_component = 0.0 if delta <= 0 else 1.0
                elif trajectory_direction == MovementDirection.STATIONARY:
                    camera_component = 0.6 if delta != 0 else 0.2

        return self._clamp_01(0.6 * direction_component + 0.4 * camera_component)

    def _compute_predictor_unhealthy(self, global_id: int, current_time: float) -> float:
        if self.prediction_manager is None:
            return 1.0

        predictor = self.prediction_manager.predictors.get(global_id)
        if predictor is None or not predictor.is_active:
            return 1.0

        max_lifetime = float(getattr(predictor, 'max_prediction_time', self.prediction_timeout))
        lifetime_ratio = (current_time - predictor.start_time) / max(max_lifetime, 1e-6)
        return self._clamp_01(lifetime_ratio)

    def _build_ghost_score(self, global_id: int, trigger_camera_id: str,
                         position: Tuple[int, int], current_time: float,
                         is_active_tracked: bool,
                         in_core_region: bool,
                         in_extended_region: bool,
                         in_blind_zone: bool) -> Tuple[float, Dict[str, Any]]:
        if self.prediction_manager is None:
            return 0.0, {'reason': 'prediction_manager_unavailable'}

        info = self.prediction_manager.vehicle_tracking_info.get(global_id)
        if info is not None:
            no_detection_sec = current_time - info.last_detection_time
        else:
            no_detection_sec = self.prediction_timeout

        effective_denominator = max(self.ghost_no_detection_threshold - self.ghost_inactive_floor_sec, 1e-6)
        time_component = self._clamp_01((no_detection_sec - self.ghost_inactive_floor_sec) / effective_denominator)
        inactive_component = 0.0 if is_active_tracked else 1.0
        predictor_component = self._compute_predictor_unhealthy(global_id, current_time)
        lane_component = self._compute_lane_inconsistency(global_id, position)
        direction_component = self._compute_direction_inconsistency(global_id, trigger_camera_id)
        region_component = 1.0 if in_core_region else (0.6 if in_extended_region else 0.0)

        w = self.ghost_score_weights
        final_score = (
            time_component * w['time_window']
            + inactive_component * w['tracking_inactive']
            + predictor_component * w['predictor_health']
            + lane_component * w['lane_consistency']
            + direction_component * w['direction_consistency']
            + region_component * w['camera_region']
        )

        if in_blind_zone:
            final_score *= 0.7

        final_score = self._clamp_01(final_score)
        evidence = {
            'camera_id': trigger_camera_id,
            'position': {'x': int(position[0]), 'y': int(position[1])},
            'in_core_region': bool(in_core_region),
            'in_extended_region': bool(in_extended_region),
            'in_blind_zone': bool(in_blind_zone),
            'time_since_last_detection_sec': float(no_detection_sec),
            'components': {
                'time_window': float(time_component),
                'tracking_inactive': float(inactive_component),
                'predictor_health': float(predictor_component),
                'lane_consistency': float(lane_component),
                'direction_consistency': float(direction_component),
                'camera_region': float(region_component),
            },
            'final_score': float(final_score),
            'thresholds': {
                'enter': self.ghost_enter_threshold,
                'exit': self.ghost_exit_threshold,
                'delete': self.ghost_delete_threshold,
            }
        }
        return final_score, evidence

    def _next_ghost_event_seq(self) -> int:
        self.ghost_event_seq += 1
        return self.ghost_event_seq

    def _record_ghost_event(self, global_id: int, current_time: float,
                          lifecycle_state: str, action: str,
                          reason: str, evidence: Optional[Dict[str, Any]] = None):
        event = {
            'seq': self._next_ghost_event_seq(),
            'timestamp': current_time,
            'vehicle_id': global_id,
            'state': lifecycle_state,
            'action': action,
            'reason': reason,
            'evidence': evidence or {}
        }
        self.ghost_event_log.append(event)

    def _clear_ghost_lifecycle(self, global_id: int, current_time: float, reason: str):
        prev_state = self.ghost_lifecycle_state.pop(global_id, None)
        if prev_state is not None:
            self._record_ghost_event(global_id, current_time, 'normal', 'clear', reason)

    def _advance_ghost_lifecycle(self, global_id: int, current_time: float,
                               reason: str, evidence: Optional[Dict[str, Any]] = None) -> str:
        lifecycle = self.ghost_lifecycle_state.get(global_id)
        if lifecycle is None:
            self.ghost_lifecycle_state[global_id] = {
                'state': 'suspect',
                'since': current_time,
                'updated': current_time,
            }
            self._record_ghost_event(global_id, current_time, 'suspect', 'enter', reason, evidence)
            return 'hold'

        state = lifecycle.get('state', 'suspect')
        lifecycle['updated'] = current_time

        if state == 'suspect':
            if current_time - lifecycle['since'] >= self.ghost_suspect_window:
                lifecycle['state'] = 'quarantine'
                lifecycle['since'] = current_time
                self._record_ghost_event(global_id, current_time, 'quarantine', 'enter', reason, evidence)
            return 'hold'

        if state == 'quarantine':
            if current_time - lifecycle['since'] >= self.ghost_quarantine_window:
                self._record_ghost_event(global_id, current_time, 'delete', 'ready', reason, evidence)
                return 'delete'
            return 'hold'

        return 'hold'

    def get_ghost_event_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self.ghost_event_log)[-limit:]
    
    def _detect_and_cleanup_ghost_vehicles(self, trigger_camera_id: str, current_time: float):
        if self.prediction_manager is None or not hasattr(self, 'id_manager'):
            return
        
        try:
            camera_region = config.CAMERA_CONFIG[trigger_camera_id]['calibration']['map_region']
            x_min, y_min, x_max, y_max = camera_region
            
            extended_x_min = max(0, x_min - 25)
            extended_x_max = min(config.MAP_WIDTH, x_max + 25)
            extended_y_min = max(0, y_min - 25)
            extended_y_max = min(config.MAP_HEIGHT, y_max + 25)
            
            ghost_vehicle_ids = []
            
            for global_id, trajectory_data in self.trajectory_manager.global_trajectories.items():
                if len(trajectory_data.trajectory) == 0:
                    continue
                
                last_position = trajectory_data.trajectory[-1]
                x, y = last_position

                in_core_region = x_min <= x <= x_max and y_min <= y <= y_max
                in_extended_region = extended_x_min <= x <= extended_x_max and extended_y_min <= y <= extended_y_max
                in_blind_zone = self._is_position_in_blind_zone(last_position)

                if not in_extended_region and not in_blind_zone:
                    continue

                if self._should_protect_blind_zone_vehicle(global_id, last_position, current_time):
                    logger.debug(f"车辆 {global_id} 位于盲区且预测健康，跳过幽灵删除")
                    continue

                is_active_tracked = self.id_manager.is_global_id_actively_tracked(
                    global_id, trigger_camera_id, current_time
                )
                if is_active_tracked:
                    self._clear_ghost_lifecycle(global_id, current_time, 'active_tracked')
                    continue

                tracking_info = self.prediction_manager.vehicle_tracking_info.get(global_id)
                if tracking_info is not None:
                    time_since_last_detection = current_time - tracking_info.last_detection_time
                    if time_since_last_detection < self.ghost_inactive_floor_sec:
                        self._clear_ghost_lifecycle(global_id, current_time, 'inactive_floor_window')
                        continue

                if not self._is_ghost_vehicle(global_id, current_time):
                    self._clear_ghost_lifecycle(global_id, current_time, 'ghost_rule_recovered')
                    continue

                score, evidence = self._build_ghost_score(
                    global_id=global_id,
                    trigger_camera_id=trigger_camera_id,
                    position=last_position,
                    current_time=current_time,
                    is_active_tracked=is_active_tracked,
                    in_core_region=in_core_region,
                    in_extended_region=in_extended_region,
                    in_blind_zone=in_blind_zone,
                )
                evidence['rule'] = 'inactive_and_ghost'

                if self.ghost_scoring_enabled:
                    if score < self.ghost_exit_threshold:
                        self._clear_ghost_lifecycle(global_id, current_time, 'score_below_exit')
                        continue

                    if score < self.ghost_enter_threshold:
                        lifecycle = self.ghost_lifecycle_state.get(global_id)
                        if lifecycle is not None:
                            lifecycle['updated'] = current_time
                            self._record_ghost_event(
                                global_id,
                                current_time,
                                lifecycle.get('state', 'suspect'),
                                'hold',
                                'score_below_enter',
                                evidence
                            )
                        continue

                action = self._advance_ghost_lifecycle(global_id, current_time, 'inactive_and_ghost', evidence)
                if action == 'delete':
                    if (not self.ghost_scoring_enabled) or score >= self.ghost_delete_threshold:
                        ghost_vehicle_ids.append(global_id)
                        logger.info(
                            f"检测到可删除幽灵车辆 {global_id} 在摄像头 {trigger_camera_id} 区域，联合评分={score:.2f}"
                        )
                    else:
                        self._record_ghost_event(
                            global_id, current_time, 'quarantine', 'hold', 'delete_score_not_enough', evidence
                        )
            
            for ghost_id in ghost_vehicle_ids:
                self._cleanup_ghost_vehicle(ghost_id, current_time)
            
            if ghost_vehicle_ids:
                logger.info(f"摄像头 {trigger_camera_id} 触发清理了 {len(ghost_vehicle_ids)} 个幽灵车辆: {ghost_vehicle_ids}")
                
        except Exception as e:
            logger.error(f"幽灵车辆检测失败: {e}")
    
    def _is_ghost_vehicle(self, global_id: int, current_time: float) -> bool:
        if self.prediction_manager is None:
            return False
            
        try:
            trajectory_data = self.trajectory_manager.global_trajectories.get(global_id)
            if trajectory_data and len(trajectory_data.trajectory) > 0:
                last_position = trajectory_data.trajectory[-1]
                if self._should_protect_blind_zone_vehicle(global_id, last_position, current_time):
                    return False

            if global_id in self.prediction_manager.vehicle_tracking_info:
                info = self.prediction_manager.vehicle_tracking_info[global_id]
                time_since_last_detection = current_time - info.last_detection_time
                
                if time_since_last_detection > self.ghost_no_detection_threshold:
                    return True
            
            vehicle_state = self.prediction_manager.get_vehicle_state(global_id)
            if vehicle_state == VehicleState.LOST:
                if global_id not in self.prediction_manager.predictors:
                    return True
                
                predictor = self.prediction_manager.predictors[global_id]
                if not predictor.is_active:
                    return True
                    
                prediction_time = current_time - predictor.start_time
                if prediction_time > self.prediction_timeout:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"判断幽灵车辆失败 {global_id}: {e}")
            return False
    
    def _cleanup_ghost_vehicle(self, ghost_id: int, current_time: Optional[float] = None):
        try:
            event_time = current_time if current_time is not None else time.time()
            self._record_ghost_event(ghost_id, event_time, 'delete', 'execute', 'cleanup_ghost_vehicle')

            if ghost_id in self.trajectory_manager.global_trajectories:
                del self.trajectory_manager.global_trajectories[ghost_id]
                logger.info(f"清理幽灵车辆 {ghost_id} 的轨迹")
            
            if ghost_id in self.trajectory_manager.global_speeds:
                del self.trajectory_manager.global_speeds[ghost_id]
            
            if ghost_id in self.trajectory_manager.global_lanes:
                del self.trajectory_manager.global_lanes[ghost_id]
            
            if ghost_id in self.vehicle_speed_cache:
                del self.vehicle_speed_cache[ghost_id]

            self.vehicle_lane_cache.pop(ghost_id, None)
            
            if self.prediction_manager and ghost_id in self.prediction_manager.predictors:
                del self.prediction_manager.predictors[ghost_id]
                logger.info(f"关闭幽灵车辆 {ghost_id} 的预测器")
            
            if self.prediction_manager:
                self.prediction_manager.state_manager.vehicle_states.pop(ghost_id, None)
                self.prediction_manager.state_manager.motion_params.pop(ghost_id, None)
                self.prediction_manager.vehicle_tracking_info.pop(ghost_id, None)
            
            if hasattr(self, 'id_manager'):
                self.id_manager.cleanup_ghost_vehicle_mapping(ghost_id)

            self._clear_ghost_lifecycle(ghost_id, event_time, 'cleanup_executed')
            
            logger.info(f"完成清理幽灵车辆 {ghost_id}")
            
        except Exception as e:
            logger.error(f"清理幽灵车辆失败 {ghost_id}: {e}")
    
    def set_id_manager(self, id_manager):
        self.id_manager = id_manager
    
    def trigger_ghost_vehicle_detection(self, camera_id: Optional[str] = None,
                                      current_time: Optional[float] = None):
        current_time = current_time if current_time is not None else time.time()
        
        if camera_id:
            self._detect_and_cleanup_ghost_vehicles(camera_id, current_time)
        else:
            for cam_id in config.CAMERA_CONFIG.keys():
                self._detect_and_cleanup_ghost_vehicles(cam_id, current_time)
    
    def get_vehicle_positions_for_speed_calculation(self) -> Dict:
        if self.prediction_manager is None:
            return {}
        return self.trajectory_manager.get_vehicle_positions_for_speed_calculation(
            self.latest_update_time, self.prediction_manager
        )
    
    def remove_stale_trajectories(self, current_time: float):
        self._cleanup_data(current_time)
    
    def _get_lane_from_position(self, position: Tuple[int, int]) -> Tuple[int, str]:
        y = position[1]
        
        for i, lane in enumerate(config.MAP_LANE_DEFINITIONS):
            y_min, y_max = lane["y_range"]
            if y_min <= y <= y_max:
                return i, lane["name"]
        
        return 0, config.MAP_LANE_DEFINITIONS[0]["name"] if config.MAP_LANE_DEFINITIONS else "unknown"
    
    def vehicle_trajectory_set(self, use_real_coordinates: bool = False) -> List[Dict[str, Any]]:
        vehicle_data = []
        
        with self.lock:
            for global_id, trajectory_data in self.trajectory_manager.global_trajectories.items():
                if len(trajectory_data.trajectory) > 0:
                    trajectory_points = list(trajectory_data.trajectory)
                    
                    lane_index, lane_name = self._get_lane_from_position(trajectory_points[-1])
                    
                    if use_real_coordinates:
                        trajectory_points = [
                            config.pixel_to_real_coordinates(x, y) 
                            for (x, y) in trajectory_points
                        ]
                    
                    v_class = self.vehicle_classes.get(global_id, 2)
                    if v_class is None:
                        v_class = 2

                    # Car(2): 5, Bus(5): 50, Truck(7): 70
                    fuel_load = 5
                    if v_class == 5:
                        fuel_load = 50
                    elif v_class == 7:
                        fuel_load = 70

                    vehicle_info = {
                        'vehicle_id': global_id,
                        'vehicle_class': v_class,
                        'trajectory_points': trajectory_points,
                        'movement_direction': trajectory_data.movement_direction.value,
                        'speed': self.vehicle_speed_cache.get(global_id, 0.0),
                        'lane_index': lane_index,
                        'lane_name': lane_name,
                        'fuel_load': fuel_load
                    }
                    vehicle_data.append(vehicle_info)
        
        return vehicle_data
