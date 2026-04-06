import scipy.spatial
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict, Tuple, Any


class CentroidTracker:
    def __init__(self, max_distance: float = 2.0, max_disappeared: int = 5):
        self.next_object_id = 0
        self.objects: Dict[int, np.ndarray] = {}  # {ID: np.array([x,y,z])}
        self.disappeared: Dict[int, int] = {}  # {ID: count}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(
        self, new_bboxes: List[Any]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """
        Updates the tracker with new detections.
        Returns:
            self.objects: Dictionary of active object IDs and their current centroids.
            assignments: Dictionary mapping bbox_index -> track_id.
        """
        assignments: Dict[int, int] = {}

        # 1. Handle no new detections
        if not new_bboxes:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects, assignments

        # Get centroids from new boxes (assuming custom objects with get_center method)
        input_centroids = np.array([b.get_center() for b in new_bboxes])

        # If no tracking objects exist, register all new detections
        if not self.objects:
            for i in range(len(input_centroids)):
                assignments[i] = self.next_object_id
                self.register(input_centroids[i])
            return self.objects, assignments

        # 2. Match Old to New
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Distance Matrix (Rows = Old, Cols = New)
        D = scipy.spatial.distance.cdist(object_centroids, input_centroids)

        # Hungarian Algorithm
        rows, cols = linear_sum_assignment(D)

        # Sets to keep track of what we have used
        used_rows = set(rows)
        used_cols = set(cols)

        # 3. Update Matched Objects
        for row, col in zip(rows, cols):
            obj_id = object_ids[row]

            if D[row, col] <= self.max_distance:
                self.objects[obj_id] = input_centroids[col]  # Update position
                self.disappeared[obj_id] = 0  # Reset counter
                assignments[col] = obj_id
            else:
                # Match rejected (too far). Free both sides.
                # DO NOT increment disappeared here, Step 4 will handle it.
                used_cols.discard(col)
                used_rows.discard(row)

        # 4. Handle UNMATCHED Old Objects (Skipped by Hungarian alg + Rejected by distance)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1

        # 5. Handle UNMATCHED New Detections (Register as new)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            assignments[col] = self.next_object_id
            self.register(input_centroids[col])

        # 6. Clean up dead tracks
        for obj_id in list(self.disappeared.keys()):
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        return self.objects, assignments
