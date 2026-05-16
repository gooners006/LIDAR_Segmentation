import scipy.spatial
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict, Tuple, Any


class CentroidTracker:
    """Frame-to-frame multi-object tracker using centroid distance matching.

    Each frame, new detections are matched to existing tracks via the
    Hungarian algorithm on a pairwise distance matrix.  Matches that
    exceed ``max_distance`` are rejected.  Unmatched detections start
    new tracks; unmatched tracks accumulate a *disappeared* counter and
    are removed after ``max_disappeared`` consecutive misses.

    Args:
        max_distance: Maximum centroid-to-centroid distance (metres) for
            a valid match.
        max_disappeared: Number of consecutive frames a track can be
            missing before it is removed.
    """

    def __init__(self, max_distance: float = 2.0, max_disappeared: int = 5):
        self.next_object_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray) -> None:
        """Create a new track for *centroid* and assign it the next ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        """Remove the track with the given *object_id*."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def _increment_disappeared(self, object_ids: list) -> None:
        """Bump the disappeared counter for each track in *object_ids*."""
        for obj_id in object_ids:
            self.disappeared[obj_id] += 1

    def _cleanup_dead_tracks(self) -> None:
        """Deregister tracks whose disappeared count exceeds the limit."""
        for obj_id in list(self.disappeared.keys()):
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

    def _match_detections(
        self, object_ids: list, object_centroids: np.ndarray,
        input_centroids: np.ndarray, assignments: Dict[int, int],
    ) -> Tuple[set, set]:
        """Match existing tracks to new detections using Hungarian algorithm.

        Computes a distance matrix between *object_centroids* and
        *input_centroids*, solves the optimal assignment, then rejects
        any match whose distance exceeds ``self.max_distance``.
        Accepted matches update the track position and populate
        *assignments* in-place.

        Args:
            object_ids: List of active track IDs (aligned with rows of
                *object_centroids*).
            object_centroids: (M, 3) array of current track positions.
            input_centroids: (N, 3) array of new detection positions.
            assignments: Dict to populate with ``{detection_index: track_id}``
                for accepted matches (modified in-place).

        Returns:
            Tuple of (used_rows, used_cols) — sets of row/column indices
            that were successfully matched.
        """
        D = scipy.spatial.distance.cdist(object_centroids, input_centroids)
        rows, cols = linear_sum_assignment(D)
        used_rows = set(rows)
        used_cols = set(cols)

        for row, col in zip(rows, cols):
            if D[row, col] <= self.max_distance:
                self.objects[object_ids[row]] = input_centroids[col]
                self.disappeared[object_ids[row]] = 0
                assignments[col] = object_ids[row]
            else:
                used_cols.discard(col)
                used_rows.discard(row)

        return used_rows, used_cols

    def update(
        self, new_bboxes: List[Any]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """Process one frame of detections and update all tracks.

        Matches *new_bboxes* to existing tracks, registers unmatched
        detections as new tracks, increments disappeared counters for
        unmatched tracks, and removes dead tracks.

        Args:
            new_bboxes: List of Open3D bounding boxes for the current
                frame (must support ``.get_center()``).

        Returns:
            Tuple of (objects, assignments).  ``objects`` is the dict of
            active track IDs to centroids.  ``assignments`` maps
            detection index to the assigned track ID.
        """
        assignments: Dict[int, int] = {}

        if not new_bboxes:
            self._increment_disappeared(list(self.disappeared.keys()))
            self._cleanup_dead_tracks()
            return self.objects, assignments

        input_centroids = np.array([b.get_center() for b in new_bboxes])

        if not self.objects:
            for i in range(len(input_centroids)):
                assignments[i] = self.next_object_id
                self.register(input_centroids[i])
            return self.objects, assignments

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        used_rows, used_cols = self._match_detections(
            object_ids, object_centroids, input_centroids, assignments,
        )

        unused_ids = [object_ids[r] for r in set(range(len(object_ids))) - used_rows]
        self._increment_disappeared(unused_ids)

        for col in set(range(len(input_centroids))) - used_cols:
            assignments[col] = self.next_object_id
            self.register(input_centroids[col])

        self._cleanup_dead_tracks()
        return self.objects, assignments
