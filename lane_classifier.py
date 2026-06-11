"""
Enhanced lane_classifier.py - Improved lane detection for vehicles
"""

import cv2
import numpy as np
import config

class LaneClassifier:
    """Lane classifier - Enhanced version with more robust detection"""
    
    def __init__(self, camera_id):
        """
        Initialize lane classifier
        Args:
            camera_id: Camera ID
        """
        self.camera_id = camera_id
        
        # Get lane definitions from config
        if camera_id in config.LANE_DEFINITIONS:
            self.lane_lines = config.LANE_DEFINITIONS[camera_id]
            print(f"Loaded lane definitions for camera {camera_id}: {len(self.lane_lines)} lanes")
            
            # Store lane boundaries for visualization
            self.lane_boundaries = []
            for i, lane in enumerate(self.lane_lines):
                self.lane_boundaries.append({
                    "name": lane["name"],
                    "polygon": lane["polygon"],
                    "color": (0, 255, 0)  # Green for visualization
                })
        else:
            # Default to 3 lanes
            print(f"Warning: No lane definitions found for camera {camera_id}, using defaults")
            h, w = 1080, 1920  # Assumed video resolution
            # Use polygons for default lanes
            self.lane_lines = [
                {"name": "lane1", "polygon": [(0, 0), (w//3, 0), (w//3, h), (0, h)]},
                {"name": "lane2", "polygon": [(w//3, 0), (2*w//3, 0), (2*w//3, h), (w//3, h)]},
                {"name": "lane3", "polygon": [(2*w//3, 0), (w, 0), (w, h), (2*w//3, h)]}
            ]
            
            # Default lane boundaries
            self.lane_boundaries = [
                {"name": "lane1", "polygon": [(0, 0), (w//3, 0), (w//3, h), (0, h)], "color": (0, 255, 0)},
                {"name": "lane2", "polygon": [(w//3, 0), (2*w//3, 0), (2*w//3, h), (w//3, h)], "color": (0, 255, 0)},
                {"name": "lane3", "polygon": [(2*w//3, 0), (w, 0), (w, h), (2*w//3, h)], "color": (0, 255, 0)}
            ]
        
        # Store lane history for smoothing
        self.vehicle_lane_history = {}  # vehicle_id -> [recent lane indices]
        self.lane_history_max = 5  # Maximum history length
    
    def determine_lane(self, bbox):
        """
        Enhanced: Use bottom center and area overlap to determine vehicle's lane
        Args:
            bbox: Vehicle bounding box [x1, y1, x2, y2]
        Returns:
            lane_index: Lane index (0, 1, 2...)
            lane_name: Lane name
        """
        if not self.lane_lines:
            return None, "out_of_lane"

        # Calculate bottom center point
        x1, y1, x2, y2 = bbox
        bottom_center = (int((x1 + x2) / 2), int(y2))
        
        # Check if bottom center is in any lane
        for lane_index, lane in enumerate(self.lane_lines):
            polygon = lane["polygon"]
            if self._point_inside_polygon(bottom_center, polygon):
                return lane_index, lane["name"]
        
        # If bottom center not in any lane, use largest overlapping area
        max_overlap = 0
        best_lane = 0
        
        # Create rectangle representation for bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        for lane_index, lane in enumerate(self.lane_lines):
            polygon = lane["polygon"]
            
            # Calculate overlap between bbox and lane polygon
            overlap = self._compute_bbox_polygon_overlap(bbox, polygon)
            
            # Normalize by bbox area
            overlap_ratio = overlap / bbox_area if bbox_area > 0 else 0
            
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                best_lane = lane_index

        if max_overlap <= 0:
            return None, "out_of_lane"

        return best_lane, self.lane_lines[best_lane]["name"]
    
    def _compute_bbox_polygon_overlap(self, bbox, polygon):
        """
        Calculate approximate overlap between bounding box and polygon
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            polygon: List of vertices [(x1, y1), (x2, y2), ...]
        Returns:
            overlap_area: Approximate overlap area
        """
        # Create a binary mask for the polygon
        x1, y1, x2, y2 = bbox
        width = max(1000, x2 + 100)
        height = max(1000, y2 + 100)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert polygon to numpy array and draw filled polygon
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Create a binary mask for the bbox
        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Compute intersection
        intersection = cv2.bitwise_and(mask, bbox_mask)
        
        # Count overlap pixels
        overlap_area = cv2.countNonZero(intersection)
        
        return overlap_area
    
    def _point_inside_polygon(self, point, polygon):
        """Check if point is inside polygon (ray casting algorithm)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def smooth_lane_assignment(self, vehicle_id, lane_index):
        """
        Smooth lane assignments to prevent flickering
        Args:
            vehicle_id: Vehicle ID
            lane_index: Current lane index
        Returns:
            smoothed_lane: Smoothed lane index
        """
        # Initialize history if needed
        if vehicle_id not in self.vehicle_lane_history:
            self.vehicle_lane_history[vehicle_id] = []
        
        # Add current lane to history
        history = self.vehicle_lane_history[vehicle_id]
        history.append(lane_index)
        
        # Limit history length
        if len(history) > self.lane_history_max:
            history = history[-self.lane_history_max:]
            self.vehicle_lane_history[vehicle_id] = history
        
        # If history too short, return current lane
        if len(history) < 3:
            return lane_index
        
        # Count occurrences of each lane
        lane_counts = {}
        for lane in history:
            if lane not in lane_counts:
                lane_counts[lane] = 0
            lane_counts[lane] += 1
        
        # Find most common lane
        most_common_lane = lane_index  # Default to current
        max_count = 0
        
        for lane, count in lane_counts.items():
            if count > max_count:
                max_count = count
                most_common_lane = lane
        
        # Only change lane if most common lane has enough support
        if max_count >= len(history) // 2:
            return most_common_lane
        
        return lane_index  # Default to current
    
    def draw_lane_boundaries(self, frame):
        """
        Draw lane boundaries on the image
        Args:
            frame: Input image
        Returns:
            vis_frame: Visualized frame with lane boundaries
        """
        vis_frame = frame.copy()
        
        # Draw each lane
        for lane in self.lane_boundaries:
            polygon = lane["polygon"]
            color = lane["color"]
            name = lane["name"]
            
            # Draw polygon outline
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], True, color, 2)
            
            # Add lane label
            # Calculate centroid
            cx = sum(p[0] for p in polygon) // len(polygon)
            cy = sum(p[1] for p in polygon) // len(polygon)
            
            cv2.putText(vis_frame, name, (cx-20, cy), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_frame
    
    def visualize_vehicle_lane(self, frame, bbox, lane_index, vehicle_id=None):
        """
        Visualize vehicle with its lane detection
        Args:
            frame: Input image
            bbox: Vehicle bounding box [x1, y1, x2, y2]
            lane_index: Detected lane index
            vehicle_id: Optional vehicle ID
        Returns:
            vis_frame: Visualized frame
        """
        vis_frame = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        lane_name = self.lane_lines[lane_index]["name"] if lane_index < len(self.lane_lines) else "Unknown"
        
        # Draw bbox with lane-specific color
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
        color = colors[lane_index % len(colors)]
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw bottom center point
        bottom_center = (int((x1 + x2) / 2), int(y2))
        cv2.circle(vis_frame, bottom_center, 5, (0, 255, 255), -1)
        
        # Add label
        label = f"Lane: {lane_name}"
        if vehicle_id is not None:
            label = f"ID: {vehicle_id}, {label}"
            
        cv2.putText(vis_frame, label, (x1, y1-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame


# Test code
if __name__ == "__main__":
    for cam_id in config.CAMERA_CONFIG:
        # Create classifier
        classifier = LaneClassifier(cam_id)
        
        # Create test image
        test_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        
        # Draw lane boundaries
        lane_img = classifier.draw_lane_boundaries(test_img)
        
        # Test some bounding boxes
        test_bboxes = [
            [456, 811, 493, 840],  # Left
            [900, 500, 1000, 1000],  # Middle
            [1700, 500, 1800, 1000]  # Right
        ]
        
        for i, bbox in enumerate(test_bboxes):
            lane_index, lane_name = classifier.determine_lane(bbox)
            
            # Test smooth lane assignment
            smoothed_lane = classifier.smooth_lane_assignment(f"test_vehicle_{i}", lane_index)
            
            # Visualize with both original and smoothed lanes
            lane_img = classifier.visualize_vehicle_lane(lane_img, bbox, smoothed_lane, f"test_{i}")
        
        # Display image
        cv2.imshow(f'Lane Classification - {cam_id}', lane_img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
