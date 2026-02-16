import scipy.spatial
from scipy.optimize import linear_sum_assignment
import numpy as np


class CentroidTracker:
    def __init__(self, max_distance=2.0, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}  # {ID: np.array([x,y,z])}
        self.disappeared = {}  # {ID: count}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0  # Initialize counter
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, new_bboxes):
        # 1. Get centroids from new boxes
        if len(new_bboxes) == 0:
            # If no new detections, mark all existing objects as 'disappeared'
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([b.get_center() for b in new_bboxes])

        # If no tracking objects exist, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
            return self.objects

        # 2. Match Old to New
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Distance Matrix (Rows = Old, Cols = New)
        D = scipy.spatial.distance.cdist(np.array(object_centroids), input_centroids)

        # Hungarian Algorithm
        rows, cols = linear_sum_assignment(D)

        # Sets to keep track of what we have used
        used_rows = set(rows)
        used_cols = set(cols)

        # 3. Update Matched Objects
        for row, col in zip(rows, cols):
            # Retrieve the id using the row index
            obj_id = object_ids[row]

            if D[row, col] < self.max_distance:
                self.objects[obj_id] = input_centroids[col]  # Update position
                self.disappeared[obj_id] = 0  # Reset counter
            else:
                self.disappeared[obj_id] += 1  # Match found but too far -> Lost

        # 4. Handle UNMATCHED Old Objects (The ones Hungarian alg skipped)
        all_rows = set(range(len(object_ids)))
        unused_rows = all_rows - used_rows

        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1

        # 5. Handle UNMATCHED New Detections (Register as new)
        all_cols = set(range(len(input_centroids)))
        unused_cols = all_cols - used_cols

        for col in unused_cols:
            self.register(input_centroids[col])

        # 6. Clean up dead tracks
        # Create a copy of keys to modify dict while iterating
        for obj_id in list(self.disappeared.keys()):
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        return self.objects
