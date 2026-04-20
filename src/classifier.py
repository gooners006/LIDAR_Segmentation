from dataclasses import dataclass
import numpy as np

CLASS_LABELS = ["pedestrian", "car", "truck", "traffic_sign", "vegetation", "unknown"]


@dataclass
class ClassificationResult:
    label: str
    confidence: float


def classify_bbox(extent: np.ndarray, center: np.ndarray) -> ClassificationResult:
    """Heuristic classification from oriented bounding box dimensions and center position.

    Args:
        extent: bbox dimensions [dim1, dim2, dim3] in metres (unsorted).
        center: bbox center [x, y, z] in local LiDAR frame.
    """
    min_d, med_d, max_d = np.sort(extent)

    # Traffic sign: very thin, elevated
    if min_d < 0.15 and max_d < 3.0 and center[2] > 0.5:
        return ClassificationResult("traffic_sign", 0.7)

    # Pedestrian: narrow, short
    if max_d < 2.0 and med_d < 1.0 and min_d < 0.8:
        return ClassificationResult("pedestrian", 0.8)

    # Car: mid-size boxy shape
    if 3.0 <= max_d <= 5.5 and 1.5 <= med_d <= 2.5 and 1.0 <= min_d <= 2.0:
        return ClassificationResult("car", 0.85)

    # Truck: long or very wide
    if max_d > 5.5 or (max_d > 4.0 and med_d > 2.0):
        return ClassificationResult("truck", 0.75)

    # Vegetation: irregular, large spread
    if max_d > 2.0 and min_d > 0.3 and med_d > 0.5:
        return ClassificationResult("vegetation", 0.6)

    return ClassificationResult("unknown", 0.0)
