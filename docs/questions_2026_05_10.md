

## 1. trong Phase 1 - Stage 6 là em đã thực hiện chưa?

Dạ rồi ạ. Stage 6 (Heuristic Classification) đã được cài đặt hoàn chỉnh trong file `src/classifier.py`. Bước này phân loại từng object được phát hiện dựa trên kích thước bounding box thành 5 lớp: **pedestrian, car, truck, traffic sign, và vegetation**.

Logic phân loại sử dụng 3 chiều của bbox (min, med, max) với các ngưỡng thủ công — ví dụ object có chiều dài 3.0–5.5 m và chiều rộng 1.5–2.5 m sẽ được phân loại là "car". Mỗi kết quả phân loại đi kèm một confidence score.

```python
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

```

Đây là phương pháp heuristic đơn giản, đóng vai trò baseline. Sau này có thể thay thế bằng learned classifier khi đã có dữ liệu object hoàn chỉnh từ Phase 2.

---

## 2.cái PCN em train cho data ShapeNet? đã test thử cho phần dữ liệu LiDAR của SemanticKiTTi chưa?

**Train trên ShapeNet:** Đã xong ạ. PCN được train trên 3,834 mẫu từ ShapeNetCore v2 (3,533 car, 939 bus, 337 motorcycle). Training chạy 100 epoch (~18 giờ trên RTX 3070 Ti), đạt F-Score 0.841 trên validation set.

**Test trên LiDAR SemanticKITTI thật:** Chưa ạ. Đây là bước tiếp theo em đang chuẩn bị làm — chạy PCN đã train lên các partial point cloud được trích xuất từ pipeline segmentation, để đánh giá khả năng transfer từ dữ liệu synthetic sang dữ liệu LiDAR thực tế (sparse, nhiễu). Em dự kiến thực hiện ngay trong thời gian tới.

---

## 3. phần Phase 2 tiếp nối task nào trong phase 1?

Phase 2 (Point Cloud Completion) là sự tiếp nối của **toàn bộ output Phase 1** — cụ thể là nó nhận các object đã được segment và track qua Stages 1–6 cộng với multi-frame tracker.

Điểm kết nối giữa 2 phase:
- Phase 1 trích xuất point cloud của từng object riêng lẻ từ LiDAR scan thô (chỉ là partial observation do occlusion và sensor sparsity)
- Phase 2 nhận các point cloud **partial** này và khôi phục lại hình dạng 3D **hoàn chỉnh** bằng PCN

Luồng xử lý: Stages 1–6 (segmentation + classification) → Tracking → **Phase 2: Completion** (Step 7, chưa kết nối vào pipeline chính).

Ngoài ra, Phase 2 còn tận dụng chế độ `--save-output` của Phase 1 để tạo cặp dữ liệu sparse/dense thực tế phục vụ fine-tuning domain adaptation.

---

## 4. stage 6 là classification thì sau khi completion của Phase 2?

Dạ không ạ. Stage 6 (Classification) thuộc **Phase 1**, và chạy **trước** Phase 2 (Completion).

Thứ tự pipeline hiện tại:

```
Phase 1:
  Stage 1: Noise Removal
  Stage 2: Voxel Downsampling
  Stage 3: Ground Removal (RANSAC)
  Stage 4: Clustering (HDBSCAN)
  Stage 5: Geometric Filtering
  Stage 6: Heuristic Classification  ← phân loại dựa trên hình dạng bbox
  Tracking (Hungarian assignment)

Phase 2 (planned Step 7):
  Point Cloud Completion (PCN)        ← khôi phục hình dạng đầy đủ từ partial scan
```

Stage 6 hiện tại dùng heuristic đơn giản (kích thước bounding box) để phân loại object. Nó không cần point cloud hoàn chỉnh — chỉ cần partial observation sau segmentation là đủ.

Tuy nhiên, một hướng cải tiến trong tương lai là thêm **bước classification thứ hai sau completion** ở Phase 2, vì hình dạng đã được khôi phục đầy đủ sẽ cung cấp đặc trưng hình học phong phú hơn cho việc phân loại. Đây sẽ là phần mở rộng của Phase 2, không thay thế Stage 6.
