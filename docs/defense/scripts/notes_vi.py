"""Vietnamese speaker notes for the support, backup and divider slides (deck_v2).

{slide number: (first line of the slide title, VI note)}. build_deck.add_vi_script puts the VI
note first and the English planning line ("○ SUPPORT ..." / "divider: ...") second; the title is
checked so a reordered deck fails loudly instead of mislabelling notes. Source references
(table labels, script names) stay in the English line only.
"""

DIVIDER = "Trang chuyển mục: bấm qua."

SUPPORT_VI = {
    3: ("I. INTRODUCTION", DIVIDER),
    6: ("1.3 LITERATURE REVIEW",
        "Không có cột độ chính xác, có chủ ý: mIoU / PQ của họ tính trên mọi lớp, còn F1 theo điểm "
        "của chúng em chỉ cho lớp xe; đó là hai phép đo khác nhau."),
    9: ("II. PROPOSED METHODOLOGY", DIVIDER),
    11: ("2.2 DETECTION PIPELINE", "Sơ đồ chuỗi phát hiện, lấy từ hình trong luận văn (chương 3)."),
    14: ("2.2.3 GEOMETRIC FILTER",
         "Riêng bước hình học: độ chính xác 0,199, độ phủ 0,773. Nếu được hỏi ngưỡng nào làm mất xe: "
         "27.718 cụm bị loại có trùng với một xe thật; 73,9% do ngưỡng thể tích tối thiểu và 12,7% do "
         "ngưỡng số điểm tối thiểu, tức là các mảnh nhỏ của những xe đã bị tách (luận văn nêu đây là "
         "suy luận)."),
    18: ("2.3.1 SYNTHETIC TRAINING PARTIALS",
         "Hình minh hoạ, không phải kết quả. Ô 1: bản quét mô phỏng do chính bộ sinh dữ liệu huấn luyện "
         "tạo ra (một góc nhìn HDL-64E, voxel 5 cm, cắt mặt đất 0,30 m). Ô 2: mô hình đầy đủ mà PCN "
         "phải dựng ra (mô hình ShapeNet có cả nội thất, nên thấy ghế). Ô 3: một bản quét thật một khung "
         "hình (xe C, slide 35). Nói: đầu vào mô phỏng có cùng góc nhìn một phía và các đường quét như "
         "bản quét thật, nên trên đường mạng gặp đúng loại đầu vào như lúc huấn luyện. Mạng nhận 256 "
         "điểm. Nếu hỏi sao không dựng đơn giản hơn: bản dựng ảnh độ sâu dày đặc lúc đầu không có đường "
         "quét, không voxel, không cắt mặt đất, nên không giống LiDAR. Huấn luyện: 100 epoch, Adam 1e-4, "
         "batch 8, loss kiểm định tốt nhất 0,1246."),
    19: ("2.3.2 REAL SCAN NORMALISATION",
         "Mã hình học thật của hệ thống chạy trên xe C (chuỗi 03, slide 18 và 35), đúng bản quét đã được "
         "hoàn thiện. Minh hoạ, không phải kết quả. 1 Kiểm tra: khớp hình chữ L 3,7 × 1,5 m, nằm trong "
         "ngưỡng (dài ít nhất 2,7 m, rộng nhiều nhất 2,3 m). 2 Xoay: chiều dài theo trục huấn luyện, trọng "
         "lực hướng lên. 3 Dịch: tâm bản quét không phải tâm xe; đẩy 0,95 − 1,52/2 = 0,19 m ra xa cảm "
         "biến (giả định bề rộng 1,9 m), 0,23 m về phía đầu không thấy, 0,25 m xuống dưới để bù phần cắt "
         "mặt đất. 4 Chỉnh tỉ lệ: chia cho khoảng cách lớn nhất / 1,137; điểm có thể nằm ra ngoài vòng "
         "tròn, đó chính là phần hiệu chỉnh. Sau đó 256 điểm vào PCN và kết quả được xoay trả lại. Cách "
         "nhớ: kiểm tra, xoay, dịch, chỉnh tỉ lệ."),
    20: ("2.3.3 GATE AND NORMALISATION CONSTANTS",
         "Phóng to bốn bước của slide 19. Nguồn gốc các con số: 1,137 là trung vị tỉ lệ giữa bán kính của "
         "bản quét một phía và bán kính thật, đo trên xe tổng hợp; 4,14 m là chiều dài xe trung vị; 1,9 m "
         "là bề rộng một xe đầy đủ; 0,25 m bù độ lệch lên trên do cắt mặt đất. Khớp hình chữ L (Zhang "
         "2017): thử các hướng hình chữ nhật cách nhau 1 độ và giữ hướng mà các điểm bám sát cạnh nhất; "
         "cạnh dài hơn là chiều dài. Trên dữ liệu thật, hướng nó cho không tốt hơn PCA; giá trị của nó là "
         "bắt được các mảnh và các cụm bị gộp. Chiều dài: phân vị 90 của các lần khớp đã qua ngưỡng trong "
         "vết, cộng 0,12 m (dùng phân vị, không dùng giá trị lớn nhất, để một lần khớp xấu không quyết "
         "định); nếu ít hơn 5 khung hình qua ngưỡng thì dùng trung vị 4,14 m."),
    21: ("III. EXPERIMENTAL EVALUATION", DIVIDER),
    22: ("3.1.1 DATASETS",
         "Bảng lấy nguyên từ luận văn. Chuỗi 08 là chuỗi kiểm định của bộ phân loại, không phải tập kiểm "
         "thử; kiểm định chéo bỏ một chuỗi (3.2.3) là cách kiểm soát rò rỉ dữ liệu."),
    23: ("3.1.2 TARGET ELIGIBILITY AND GROUND TRUTH",
         "Xe được tính khi còn ít nhất 10 điểm sau tiền xử lý; 25,5% số xe được gán nhãn bị loại. Độ phủ "
         "trên tất cả xe được gán nhãn khoảng 0,55, so với 0,732 trên các xe đủ điều kiện."),
    24: ("3.2 EXPERIMENTAL SETUP", DIVIDER),
    25: ("3.2.1 BASELINES",
         "Vì sao không so trực tiếp với các mạng có giám sát: họ báo cáo mIoU trên mọi lớp / chất lượng "
         "panoptic và huấn luyện bằng nhãn dày đặc từng điểm; chúng em báo cáo F1 theo điểm chỉ cho lớp "
         "xe. Đặt chung một cột là so những con số mà độ đo không cho phép so; khoảng cách về độ chính xác "
         "là có thật và được nêu trong luận văn. Bước hoàn thiện không có mốc so sánh bên ngoài trên dữ "
         "liệu thật, nên được so với chính đầu vào của nó (theo cặp), một tham chiếu dày đặc (mức trần) "
         "và PoinTr."),
    27: ("3.2.3 DATA SPLIT: LEAVE-ONE-SEQUENCE-OUT",
         "Fold 08 và bộ phân loại cuối cùng dùng chung dữ liệu, kiến trúc và chế độ 15 epoch; chỉ khác ở "
         "epoch được giữ lại. Gộp chung (pooled) là cộng dồn số đếm (độ phủ 0,717); trung bình đơn giản "
         "của 11 fold là 0,632 ± 0,119, bị kéo xuống bởi hai chuỗi nhỏ, thưa là 01 và 04. Fold chuỗi 00: "
         "P 0,921 / R 0,758 / F1 0,831."),
    28: ("3.3 DETECTION RESULTS", DIVIDER),
    30: ("3.3.2 DETECTIONS ON A REAL FRAME",
         "Hình lấy nguyên từ luận văn. Dương tính giả khoảng 0,6 mỗi khung hình (2.513 / 4.071)."),
    32: ("3.3.4 ACCURACY VS RANGE",
         "Bảng lấy nguyên từ luận văn; tỉ lệ bị tách lấy từ phần phân tích tách cụm (chương 4). Mẫu cách "
         "20 khung hình, tính theo từng khung, tắt bộ lọc theo vết (lấy mẫu cách quãng làm hỏng bộ theo "
         "vết), nên độ phủ gộp 0,671 không phải con số chính 0,732. IoU trung bình của các xe được ghép "
         "tăng theo khoảng cách: những xe ở xa tìm được là các cụm gọn, nguyên một khối."),
    33: ("3.3.5 DETECTION RESULT ANALYSIS",
         "Cả bộ phân loại lẫn bộ theo vết đều không tạo ra phát hiện mới, nên độ phủ không bao giờ vượt "
         "0,773 của bước hình học. Độ chính xác đã bão hoà (khoảng 0,6 dương tính giả mỗi khung hình), "
         "độ phủ là thứ bị giới hạn. 33,7% số xe bị tách ở lưới 0,10 m."),
    34: ("3.4 COMPLETION RESULTS", DIVIDER),
    37: ("3.4.3 SYNTHETIC FIDELITY",
         "Xe minh hoạ lấy từ tập kiểm định của PCN (chưa từng được huấn luyện trên nó), chọn theo quy tắc "
         "cố định (xe có Chamfer trung vị trong tập đánh giá), không chọn bằng mắt; Chamfer 0,16 m / F "
         "0,77 là của riêng xe đó. Khoảng cách Chamfer: trung bình khoảng cách tới điểm gần nhất theo cả "
         "hai chiều, cộng lại (mét). F-score @ 0,1 m: trung bình điều hoà của tỉ lệ điểm đầu ra nằm trong "
         "0,1 m của mô hình thật và tỉ lệ mô hình thật nằm trong 0,1 m của đầu ra. Loss kiểm định tốt "
         "nhất 0,1246 (Chamfer thô + 0,5 × Chamfer mịn, đơn vị đã chuẩn hoá; không cùng con số với "
         "0,16 m)."),
    38: ("3.4.4 PLAUSIBILITY ROBUSTNESS",
         "Bảng lấy nguyên từ luận văn. McNemar: 1.615 xe chuyển sang được chấp nhận so với 115 theo chiều "
         "ngược lại; Wilcoxon trên xác suất là xe: p ≈ 0. Xe đang chạy: 0,837 / 0,801, N = 602; lật hướng "
         "xe chỉ thay đổi 0,009. Nhóm xe đỗ, N = 2.220: tỉ lệ chấp nhận sau hoàn thiện 0,770, xác suất là "
         "xe 0,730 so với mức trần 0,728, tức khoảng 96% khoảng cách từ bản quét gốc tới mức trần. Xe "
         "chạy hay đỗ: phân theo độ dịch chuyển ròng của tâm vết. PCN chưa từng thấy chuỗi thật nào, nên "
         "kết quả ổn định trên cả mười một chuỗi là một kết quả về khả năng tổng quát. Vì sao không chỉ "
         "do mật độ: tham chiếu dày đặc vừa dày vừa đúng hình dạng, nên đạt ngang nó nghĩa là hình dạng "
         "hoàn thiện giống xe, không chỉ là dày điểm."),
    39: ("3.4.5 COMPLETION RESULT ANALYSIS",
         "Bảng lấy nguyên từ luận văn (bỏ cột UHD p95: 0,000 / 0,000 / 0,183). UCD: với mỗi điểm của bản "
         "quét, khoảng cách tới điểm gần nhất của đám mây, lấy trung bình; UHD: khoảng cách lớn nhất trong "
         "số đó. Chiều tính từ bản quét sang đám mây, trung vị theo từng xe, như Chen và cộng sự và P2C "
         "áp dụng. Bản phản chiếu = bản quét cộng bản sao đối xứng của nó: nó chứa bản quét nên cũng được "
         "điểm 0. Đó là lý do các nghiên cứu hoàn thiện không có dữ liệu cặp luôn dùng kèm các độ đo này "
         "với một thành phần đánh giá tính hợp lý."),
    40: ("3.5 RUNTIME EVALUATION", DIVIDER),
    41: ("3.5.1 RUNTIME BY STAGE",
         "Bảng lấy nguyên từ luận văn (RTX 3070 Ti / Ryzen 7 7800X3D). Bộ phân loại 33,7 ms cho khoảng 45 "
         "cụm, khoảng 0,75 ms mỗi cụm. Bước hoàn thiện thêm khoảng 13 ms cho mỗi xe được hoàn thiện, nằm "
         "ngoài ngân sách này."),
    42: ("3.5.2 RUNTIME OPTIMISATION",
         "RANSAC giảm từ 1.000 xuống 300 vòng lặp; thời gian trên toàn chuỗi từ 921,3 xuống 623,5 ms "
         "(650,6 ms trên mẫu cách 20 khung hình). Đám mây sau khi bỏ mặt đất đã thưa, nên lưới thô hơn nén "
         "thêm không đáng kể; HDBSCAN viết bằng Python vẫn có mức sàn khoảng 185 ms."),
    43: ("3.6 ABLATION STUDY", DIVIDER),
    44: ("3.6.1 STAGE ABLATION",
         "Bảng lấy nguyên từ luận văn. Cùng số liệu với biểu đồ ở slide 29, thêm IoU trung bình và số đếm "
         "thô. Số đếm là số lần ghép theo từng khung hình trên toàn chuỗi, không phải số xe riêng biệt."),
    45: ("3.6.2 CLUSTERING ALTERNATIVES AND RESOLUTION",
         "Bỏ cột IoU trung bình của bảng phân cụm. Nếu hỏi 'sao không dùng lưới thô hơn nữa, ví dụ 0,20 m?': "
         "số lần gộp tăng khi lưới thô hơn (56 → 284), nên chúng em dự đoán sẽ có thêm xe bị gộp; 0,20 m "
         "CHƯA được thử, nói 'dự đoán', không nói 'đã thấy'. Nếu hỏi 'có tinh chỉnh bán kính của DBSCAN "
         "không?': có, đã quét giá trị epsilon; bán kính cố định tốt nhất (Euclid, 0,4 m) đạt F1 0,800 "
         "trên chuỗi 00, vẫn thấp hơn 0,835 của HDBSCAN."),
    46: ("3.6.3 IOU-THRESHOLD SENSITIVITY",
         "Bảng lấy nguyên từ luận văn. Ngưỡng 0,3 là giá trị mặc định của bộ đánh giá từ đầu, không tinh "
         "chỉnh theo kết quả; điểm chính là kết luận vẫn đúng ở 0,25 và 0,5 (H2, qa_bank.md). Đừng nói "
         "'với hộp bao, ngưỡng là 50%': bộ chuẩn hộp bao cho xe của KITTI dùng 0,7. Đã đăng ký trước: nếu "
         "kết quả nhạy với ngưỡng thì sẽ được báo cáo như một phát hiện, không chỉnh để che đi."),
    47: ("3.6.4 SYNTHETIC PRETRAINING",
         "Macro-F1 0,9285 so với 0,9225; F1 của cả hệ thống giống nhau tới ba chữ số thập phân. Đừng trộn "
         "độ đo: 0,9285 là macro-F1 (trung bình F1 của lớp xe và lớp không phải xe), 0,885 là F1 của lớp "
         "xe. Vì sao vẫn giữ trong luận văn: nó thay đổi công thức huấn luyện, nên được báo cáo như một "
         "kết quả âm."),
    48: ("3.6.5 COMPLETION ABLATIONS",
         "Mỗi dòng một thí nghiệm. Chuẩn hoá: cùng một checkpoint, chỉ khác hệ toạ độ đầu vào, nên chênh "
         "lệch giữa khối mờ và chiếc xe là do chuẩn hoá. Tinh chỉnh: các khối mờ lúc đầu bị cho là do khác "
         "biệt miền dữ liệu, nên PCN được tinh chỉnh trên bản quét mô phỏng giống LiDAR hơn (tăng cường "
         "nhiễu, Velodyne ảo): vẫn ra khối mờ, vì nguyên nhân là hệ toạ độ đầu vào. Tinh chỉnh trên dữ "
         "liệu thật chưa bao giờ được chạy: đó là hướng phát triển (slide 52); đừng nói là đã thử. Bước "
         "kiểm tra hình dạng: cái giá là một số xe dùng được bị từ chối. Chiều dài: những xe dài nhất vẫn "
         "còn hụt một chút ở đầu xa (chương 5). PoinTr: cùng các xe tổng hợp giữ lại; khoảng cách còn lại "
         "trên dữ liệu thật là do ước lượng tâm, không mạng nào sửa được."),
    49: ("IV. CONCLUSION & FUTURE WORK", DIVIDER),
    54: ("LIDAR POINT CLOUD PROPERTIES",
         "Slide dự phòng về đặc tính đám mây điểm LiDAR (chương 2, bảng thông số cảm biến)."),
    55: ("HOW HDBSCAN WORKS",
         "Dự phòng cho câu 'HDBSCAN hoạt động thế nào?'. Ví dụ đồ chơi 2D, dùng thư viện hdbscan thật với "
         "tham số của chúng em (min_samples 5, min_cluster_size 10); KHÔNG phải kết quả của luận văn. Nói: "
         "(1) mỗi điểm đo độ đông đúc quanh nó: khoảng cách tới điểm gần thứ 5 (tính cả chính nó); "
         "(2) HDBSCAN xét mọi bán kính cùng lúc: ở bán kính nhỏ chỉ chiếc xe gần, dày đặc thành cụm; ở bán "
         "kính vừa cả hai xe thành cụm; ở bán kính lớn hai xe dính vào nhau; (3) khi thu nhỏ bán kính, nó "
         "giữ nguyên một nhóm nếu nhóm đó tồn tại lâu hơn các mảnh của nó. Mỗi xe chỉ được so với các mảnh "
         "của chính nó, không bao giờ so với xe kia, nên cả xe gần dày đặc lẫn xe xa thưa thớt đều được "
         "giữ; DBSCAN bán kính cố định thì buộc phải chọn một cột. Số liệu ví dụ: xe gần 335,7 so với các "
         "mảnh 19,2 → giữ nguyên; xe xa 24,1, không có mảnh → giữ. Nếu bị hỏi kỹ: 'bán kính' là khoảng cách "
         "tiếp cận lẫn nhau max(core a, core b, khoảng cách a-b), và 'lâu hơn' là độ ổn định: thời gian tồn "
         "tại đo theo 1/bán kính, cộng trên các điểm; nhóm dưới 10 điểm là nhiễu."),
    56: ("REPAIR STRATEGIES AND WHY THEY FAIL",
         "Dự phòng cho câu 'có thử sửa việc tách xe không?'. Bảng lấy nguyên từ luận văn. Mốc so sánh là "
         "chuỗi 00 trước khi nâng cấp (lưới 0,05 m), nên F1 0,844 không phải hệ thống cuối cùng. Giới hạn "
         "chung: phân cụm theo mật độ không có khái niệm 'vật thể', nên không quy tắc khoảng cách hay kích "
         "thước nào phân biệt được 'nối lại hai mảnh của một xe' với 'giữ hai xe sát nhau tách riêng'. "
         "Lưới 0,10 m hiệu quả vì nó lấp khe bên trong một xe trước khi lấp khe giữa hai xe."),
    57: ("NORMALISATION ABLATION ON A REAL CAR",
         "Dự phòng cho câu 'vì sao cần bốn bước hình học?' hoặc 'sao không tinh chỉnh trên dữ liệu thật?'. "
         "Hình: xe C (cố định trước), cùng checkpoint pcn_kitti_best ở cả hai bên; trái = cách làm ban đầu "
         "(tâm và bán kính của chính bản quét, PCA 3D, trục dài nhất làm trục X), phải = đầu ra của hệ "
         "thống. Đầu ra bên trái rộng khoảng 5,1 × 3,5 × 2,6 m, bên phải 3,9 × 1,8 × 1,7 m (minh hoạ, không "
         "phải số liệu luận văn). Nói: mạng chưa bao giờ là vấn đề; hệ toạ độ đầu vào mới là vấn đề. Các "
         "khối mờ lúc đầu bị cho là do khác biệt giữa dữ liệu tổng hợp và dữ liệu thật, nhưng huấn luyện lại "
         "trên bản quét mô phỏng chân thực hơn không giúp được; đưa bản quét về đúng hệ toạ độ huấn luyện "
         "mới giúp được."),
    58: ("IS THE JUDGE INDEPENDENT?",
         "Dự phòng cho mọi thắc mắc về slide 36. Chen và cộng sự (2020) cũng dùng bộ đánh giá huấn luyện "
         "trên ModelNet40 vì cùng lý do. Việc đăng ký trước và ngưỡng 0,88 được viết ra trước khi chấm "
         "điểm. ModelNet40 không có lớp xe tải, xe buýt hay xe van, nên lớp 'xe' không bị chia với các lớp "
         "phương tiện gần giống."),
    59: ("FIDELITY METRICS: UCD AND UHD",
         "Dự phòng cho câu 'sao không dùng các độ đo chuẩn?'. Hình vẽ lại dạng ví dụ đồ chơi, các con số là "
         "của ví dụ. Chen và cộng sự 2020 (UHD) và P2C 2023 (UCD) dùng các độ đo này trên xe KITTI. Luận "
         "văn có báo cáo chúng, kèm lưu ý rằng chúng thưởng cho việc hoàn thiện thiếu."),
    60: ("WORKED EXAMPLE: IOU MATCHING",
         "Dự phòng cho câu 'chính xác thì một phát hiện được ghép thế nào?'. Ví dụ là minh hoạ. Một xe bị "
         "tách bị tính hai lần: một xe bị bỏ sót và thường thêm một phát hiện sai. Số đếm tính theo từng "
         "khung hình, cộng gộp (micro-average)."),
    61: ("PIPELINE CONFIGURATION",
         "Dự phòng cho câu 'dùng những tham số nào?'. Bảng lấy nguyên từ luận văn. Cố định trước khi đánh "
         "giá; không tham số nào được chỉnh theo kết quả báo cáo. Các hằng số của bước hoàn thiện: slide 20."),
}
