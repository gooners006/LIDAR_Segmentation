# Q&A Bank

Committee: general computer-science knowledge, no LiDAR or 3D-vision specialist (`defense-logistics`).
Their questions will follow what a generalist notices. Part 1 plants that curiosity on purpose.

---

## Part 1 — Planted hooks (2026-10-18)

**Rules.** A hook is one sentence in the talk that states a result with a number and **stops**,
leaving the obvious "why / how?" unanswered. Each hook leads to a slide with a strong, visual answer.
- Five hooks, about one per section; more makes the talk sound evasive.
- Say the line, pause a beat, move on. Never say "ask me later" or "I'll explain in the questions".
- Nothing is hidden: every hook's answer is in the thesis; the talk just doesn't spend its seconds there.
- Jump to the answer slide by number (type the number + Enter in slideshow).

| # | Slide (when) | Planted line | Question it invites | Answer slide |
|---|---|---|---|---|
| H1 | 17, 2.3 Completion | "At first the network produced only blobs. The fix was not more training." | "So what was the fix?" | **57** |
| H2 | 26, 3.2.2 Metrics | "A detection counts when it shares at least 30% of its points with a real car." | "Why 30%? Isn't that lenient?" | **46** |
| H3 | 31, 3.3.3 Recall | "We tried six ways to repair the splitting; only one worked." | "What did you try?" | **56** |
| H4 | 36, 3.4.2 Judge | "We ask an independent judge, a separate program trained on different 3D models." | "How can one program judge another? Is it fair?" | **58** (59) |
| H5 | 51, 4.2 Limitations | "623.5 ms per frame, offline by design." | "Could it run in real time?" | **41–42** |

### H1 — "So what was the fix?" → slide 57

- EN (25 s): "The network expects a car in the same position, direction and size it saw in training.
  Our first version put each scan in a generic frame: its own centre, its own size, and its longest
  direction as the main axis, which isn't how the network was trained. So even a good network got a
  confusing input and returned a blob. Here is the same network on the same scan: on the left the old
  frame, on the right our four geometry steps. The network was never the problem; the input frame was.
  That's also why retraining on more realistic simulated scans didn't help."
- VI: "Mạng nơ-ron kỳ vọng chiếc xe nằm ở đúng vị trí, hướng và kích thước như lúc huấn luyện. Phiên
  bản đầu tiên của chúng tôi đặt mỗi bản quét vào một hệ toạ độ chung chung: lấy tâm và kích thước của
  chính bản quét, và lấy hướng dài nhất làm trục chính, không giống cách mạng được huấn luyện. Vì vậy
  ngay cả một mạng tốt cũng nhận đầu vào khó hiểu và trả về một khối mờ. Đây là cùng một mạng, cùng một
  bản quét: bên trái là hệ toạ độ cũ, bên phải là bốn bước hình học của chúng tôi. Vấn đề chưa bao giờ
  nằm ở mạng, mà ở hệ toạ độ đầu vào. Đó cũng là lý do tinh chỉnh trên dữ liệu thật không giúp được gì."
- Follow-up "why not just train it on rotated cars?": "We could, but then the network has to learn
  orientation as well as shape from the same data. Putting the scan in a fixed frame is free and exact,
  and it worked."

### H2 — "Why 30%?" → slide 46

- EN (20 s): "Because we compare points, not boxes. With boxes, 50% is the usual rule; with points,
  30% already means the detection and the car share a large part of their points. We checked the
  strict 50% as well: F1 only drops from 0.812 to 0.788. And the detections we match overlap their car
  by 0.91 on average, far above either threshold."
- VI: "Vì chúng tôi so sánh theo điểm, không theo hộp bao. Với hộp bao, ngưỡng thường dùng là 50%; còn
  với điểm, 30% đã có nghĩa là phát hiện và chiếc xe chia sẻ phần lớn số điểm. Chúng tôi cũng đã kiểm tra
  ngưỡng chặt 50%: F1 chỉ giảm từ 0,812 xuống 0,788. Và các phát hiện được ghép cặp trùng với xe trung
  bình 0,91, cao hơn hẳn cả hai ngưỡng."
- Pre-registered: a threshold-sensitive result would have been reported as a finding, not tuned away.

### H3 — "What did you try?" → slide 56

- EN (30 s): "Six things: a bigger minimum cluster size, merging nearby pieces, settings that change
  with distance, a bird's-eye-view clusterer, stacking several frames, and lower size thresholds. Each
  failed for a clear reason. The bigger minimum helped on sequence 00 but hurt on 08, because it also
  erases small, distant cars. Merging nearby pieces also merges cars with walls and poles, so precision
  fell. The one that worked was clustering on a 10-centimetre grid instead of 5: recall 0.710 to 0.732.
  It closes the small gaps inside a car before it closes the gaps between two cars."
- VI: "Sáu cách: tăng kích thước cụm tối thiểu, gộp các mảnh ở gần nhau, tham số thay đổi theo khoảng
  cách, phân cụm nhìn từ trên xuống, chồng nhiều khung hình, và hạ ngưỡng kích thước. Mỗi cách thất bại
  vì một lý do rõ ràng. Tăng kích thước tối thiểu giúp ở chuỗi 00 nhưng làm kém đi ở chuỗi 08, vì nó cũng
  xoá mất các xe nhỏ ở xa. Gộp các mảnh gần nhau cũng gộp xe với tường và cột, nên độ chính xác giảm. Cách
  duy nhất hiệu quả là phân cụm trên lưới 10 cm thay vì 5 cm: độ phủ tăng từ 0,710 lên 0,732. Nó lấp
  những khe nhỏ bên trong một chiếc xe trước khi lấp khe giữa hai chiếc xe."
- Follow-up "why not an even coarser grid?": merges grow as the grid coarsens (56 → 284 cars fused
  at 0.10 m); 0.20 m was not tested, so say "we expect", not "we found".

### H4 — "How can one program judge another? Is it fair?" → slide 58 (59)

- EN (30 s): "Three reasons it's fair. It never saw our network's training data: it learned from a
  different collection of 3D models. It isn't our own classifier. And it isn't easy to please: it
  rejects three in four raw scans. It also accepts dense, real multi-frame scans of parked cars 81% of
  the time, so it does recognise real cars when the shape is complete. We fixed its pass mark, 0.88 test
  accuracy, and what each result would mean before scoring anything; it reached 0.92."
- VI: "Có ba lý do để tin là công bằng. Nó chưa từng thấy dữ liệu huấn luyện của mạng chúng tôi: nó học
  từ một bộ mô hình 3D khác. Nó không phải bộ phân loại của chính chúng tôi. Và nó không dễ dãi: nó từ chối
  ba trên bốn bản quét gốc. Nó cũng nhận ra 81% các bản quét dày đặc nhiều khung hình của xe đỗ, tức là nó
  nhận ra xe thật khi hình dạng đầy đủ. Chúng tôi đặt trước ngưỡng đạt, độ chính xác 0,88, và ý nghĩa của
  mỗi kết quả trước khi chấm điểm; bộ đánh giá đạt 0,92."
- If they ask "why not just measure the distance?": slide 59, "copying the scan scores a perfect zero".
- Don't volunteer: judge heading alignment (S6 notes trap #10; the aligned re-score is the backup
  answer only if pressed on orientation).

### H5 — "Could it run in real time?" → slides 41–42

- EN (25 s): "Not today: about 0.6 seconds per scan, 1.6 frames per second. Real time was out of scope.
  Most of the time is clustering, and that's the Python implementation, not the method: a native or GPU
  HDBSCAN is the direct path, and it's our first future-work item. Swapping in a simpler, faster
  clusterer isn't free: we measured it, and it loses about ten points of recall."
- VI: "Hiện tại thì chưa: khoảng 0,6 giây mỗi lần quét, 1,6 khung hình mỗi giây. Thời gian thực nằm ngoài
  phạm vi đề tài. Phần lớn thời gian là bước phân cụm, và đó là do cài đặt bằng Python chứ không phải do
  phương pháp: một bản HDBSCAN viết bằng mã gốc hoặc chạy trên GPU là hướng đi trực tiếp, và đó là việc
  đầu tiên trong hướng phát triển. Thay bằng một thuật toán phân cụm đơn giản hơn, nhanh hơn thì không
  miễn phí: chúng tôi đã đo, và nó mất khoảng mười điểm độ phủ."
- Numbers: tab:runtime clustering 370.4 ms (59.4%), Python HDBSCAN floor ~185 ms; tab:clustering seq 08
  recall 0.671 (HDBSCAN) vs 0.565 (DBSCAN) / 0.576 (Euclidean).

---

## Part 2 — Do not open (no hooks near these; answers only if asked)

All in `docs/project_state.md` → Known Thesis Weak Spots, with prepared answers in the S1–S6 notes.
- Judge input heading / alignment (S6 trap #10).
- PCN training categories; roll augmentation (S5 traps #9, #13).
- How many car tracks get completed; the length fallback rate (S6 trap #15; S5).
- Synthetic 0.16 m placement (S6 trap #6).
- **Real-data fine-tuning.** ch4 says it was tried and failed; the repo shows it never was (the
  fine-tuning runs, findings #16/#17/#19, 2026-05-28 to 06-02, used LiDAR-like *simulated* scans; no
  real-data checkpoint exists). The deck now says "fine-tune on noisier simulated scans" (slide 48) and lists
  real-data fine-tuning as future work (slide 52), which is consistent. If asked "the thesis says you
  fine-tuned on real data": "The fine-tuning we ran was on more realistic simulated scans, and it was
  done while the input frame was still wrong, so it couldn't help. Fine-tuning on real data is the next
  step, with the test cars kept out of training." Don't volunteer.

---

## Part 3 — Questions they will ask anyway (2026-10-18)

What a generalist will ask without any hook, in rough order of likelihood. Each: a 15–25 s answer,
the slide to jump to, and the trap behind it. Numbers are from the thesis unless tagged.

### A. Positioning

**Q1. "How does this compare with deep learning? Is it better?"** → slides 5, 6 (25)
- EN: "On accuracy, no, and we don't claim it. The best deep networks score higher, but they need a
  label on every point to train and they score all classes on another test set, so the numbers aren't
  on the same scale. What we offer is different: no hand-labelled point clouds, every stage can be
  inspected, and when something fails we can find the cause. That's how we found that recall is
  limited by clustering, not by the classifier."
- VI: "Về độ chính xác thì không, và chúng tôi không khẳng định điều đó. Các mạng học sâu tốt nhất đạt
  điểm cao hơn, nhưng chúng cần nhãn cho từng điểm để huấn luyện và chấm điểm trên tất cả các lớp với một
  tập kiểm thử khác, nên các con số không cùng thang đo. Điều chúng tôi mang lại là khác: không cần đám mây
  điểm gán nhãn thủ công, mọi bước đều kiểm tra được, và khi có lỗi thì tìm được nguyên nhân. Nhờ vậy chúng
  tôi phát hiện độ phủ bị giới hạn bởi bước phân cụm, không phải bởi bộ phân loại."
- Trap: don't put their mIoU (52.2–67.8) next to our F1 0.812 as if comparable [ch2 §positioning].

**Q2. "Why build a modular pipeline at all, if end-to-end is more accurate?"** → slide 5
- EN: "Two costs of end-to-end: labels and opacity. Labelling every point of a dataset like
  SemanticKITTI is very expensive, and a network spreads each decision over millions of weights, so a
  missed car is hard to explain. A modular pipeline needs no new labels and every stage can be tested
  on its own. The thesis asks how far that approach goes."
- VI: "Phương pháp đầu-cuối có hai cái giá: nhãn và tính khó giải thích. Gán nhãn từng điểm cho một bộ dữ
  liệu như SemanticKITTI rất tốn kém, và một mạng nơ-ron phân tán mỗi quyết định qua hàng triệu trọng số,
  nên khó giải thích vì sao một chiếc xe bị bỏ sót. Hệ thống dạng mô-đun không cần thêm nhãn mới và từng
  bước có thể kiểm tra riêng. Luận văn này đặt câu hỏi phương pháp đó đi được bao xa."

**Q3. "What is this used for in practice?"** → slides 4, 52
- EN: "Self-driving perception: knowing where other cars are, how big they are and which way they
  face. A scanner only sees the near side of each car; completion estimates the hidden side, which is
  what you need for an accurate box around the car. Testing that downstream benefit is our next step."
- VI: "Hệ thống nhận thức cho xe tự lái: biết các xe khác ở đâu, lớn cỡ nào và quay về hướng nào. Cảm
  biến chỉ thấy phía gần của mỗi chiếc xe; bước hoàn thiện ước lượng phía bị che, là thứ cần để có hộp
  bao chính xác quanh xe. Kiểm chứng lợi ích đó cho các bước sau là hướng tiếp theo của chúng tôi."
- Trap: don't say completion already improves detection; it isn't measured (ch5 future work).

**Q4. "Why only cars?"**
- EN: "Cars are the most common road user in the data and the class the completion step needs full
  3D models for. Detection is a binary car / not-car decision, which keeps the classifier small and
  its labels cheap to mine. Other classes would follow the same recipe, with their own models and
  priors; that's not tested here."
- VI: "Xe ô tô là đối tượng phổ biến nhất trong dữ liệu và là lớp mà bước hoàn thiện cần mô hình 3D đầy
  đủ. Bước phát hiện là quyết định nhị phân xe / không phải xe, giúp bộ phân loại nhỏ gọn và nhãn dễ khai
  thác tự động. Các lớp khác có thể theo cùng cách làm, với mô hình và giá trị tiên nghiệm riêng; điều đó
  chưa được kiểm chứng ở đây."
- Trap: the 1.9 m width prior, 4.14 m length and gate thresholds are car-specific; say so if asked
  about trucks.

### B. Data and labels

**Q5. "You say no annotation, but you used SemanticKITTI's labels?"** → slide 15
- EN: "Right, that's why we say zero *additional* manual annotation. The labels already exist; we use
  them automatically to tell which of our clusters are cars, and train the classifier on that. Nobody
  drew a single new label. The geometric stages use no labels at all."
- VI: "Đúng vậy, vì thế chúng tôi nói là không cần *thêm* nhãn thủ công. Các nhãn đã có sẵn; chúng tôi
  dùng chúng một cách tự động để biết cụm nào là xe, rồi huấn luyện bộ phân loại trên đó. Không ai phải
  vẽ thêm một nhãn mới nào. Các bước hình học hoàn toàn không dùng nhãn."
- Trap: never say "annotation-free" or "unsupervised" for the whole system.

**Q6. "Did you test on data you trained on?"** → slide 27
- EN: "No. The classifier is trained on ten sequences and tested on the eleventh, and we repeat that
  for all eleven: precision 0.888, recall 0.717, F1 0.793, within about two points of the headline.
  The completion network never saw any real scan at all."
- VI: "Không. Bộ phân loại được huấn luyện trên mười chuỗi và kiểm thử trên chuỗi thứ mười một, lặp lại
  cho cả mười một chuỗi: độ chính xác 0,888, độ phủ 0,717, F1 0,793, chênh khoảng hai điểm so với kết quả
  chính. Mạng hoàn thiện thì chưa từng thấy bất kỳ bản quét thật nào."
- Trap: seq 08 is the validation sequence (checkpoint picked on it); the blind fold-08 is 0.888 /
  0.736 / 0.805. Say it if pressed (S4 notes).

**Q7. "Why train completion on synthetic cars?"** → slides 17–18
- EN: "Because no real scan ever shows a whole car: the far side is always hidden, so there's nothing
  real to learn the hidden side from. 3D car models give the complete shape, and we simulate what our
  own sensor would see of them, so the training inputs look like real scans."
- VI: "Vì không có bản quét thật nào thấy được toàn bộ chiếc xe: phía xa luôn bị che, nên không có dữ
  liệu thật nào để học phía bị che. Các mô hình 3D của xe cho hình dạng đầy đủ, và chúng tôi mô phỏng
  những gì cảm biến của mình sẽ thấy, để dữ liệu huấn luyện trông giống bản quét thật."
- Trap: say "3D vehicle models", not "only cars" (S5 trap #9).

**Q8. "KITTI is from 2012. Isn't it outdated?"**
- EN: "It's still the standard benchmark with point-level labels for LiDAR, so results are comparable
  and checkable. Newer datasets use different sensors; carrying the pipeline over would mean
  re-simulating that sensor for completion and re-checking the thresholds. That's not tested."
- VI: "Đây vẫn là bộ chuẩn phổ biến có nhãn ở mức điểm cho LiDAR, nên kết quả có thể so sánh và kiểm
  chứng. Các bộ dữ liệu mới hơn dùng cảm biến khác; chuyển hệ thống sang đó cần mô phỏng lại cảm biến cho
  bước hoàn thiện và kiểm tra lại các ngưỡng. Điều đó chưa được thử."

### C. Method

**Q9. "Why HDBSCAN and not a simpler clustering?"** → slides 13, 45 (55)
- EN: "Point density changes a lot with distance: a near car has thousands of points, a far one a
  few dozen. A fixed-radius method has to pick one distance and fails at the other. HDBSCAN adapts to
  density. We measured the faster fixed-radius ones: about ten points less recall."
- VI: "Mật độ điểm thay đổi rất nhiều theo khoảng cách: xe ở gần có hàng nghìn điểm, xe ở xa chỉ vài chục.
  Phương pháp bán kính cố định phải chọn một khoảng cách và sẽ thất bại ở phía còn lại. HDBSCAN thích ứng
  với mật độ. Chúng tôi đã đo các phương pháp bán kính cố định nhanh hơn: độ phủ thấp hơn khoảng mười điểm."

**Q10. "What about two cars parked close together?"** → slide 45
- EN: "Sometimes they merge into one cluster, but rarely: 0.8% of cars at our grid size. Splitting one
  car into pieces is the much bigger problem, a third of cars. For completion, merged pairs are caught
  by the size check and skipped."
- VI: "Đôi khi chúng bị gộp thành một cụm, nhưng hiếm: 0,8% số xe ở kích thước lưới của chúng tôi. Việc
  một chiếc xe bị tách thành nhiều mảnh là vấn đề lớn hơn nhiều, khoảng một phần ba số xe. Với bước hoàn
  thiện, các cặp bị gộp bị phát hiện bởi bước kiểm tra kích thước và được bỏ qua."

**Q11. "Why is tracking needed?"** → slide 16
- EN: "The classifier decides frame by frame and sometimes flips. Tracking lets us decide once per
  object: a car the classifier misses in one frame is kept, a bush it calls a car once is dropped. On
  seq 08 that wins back 1,968 correct detections and removes 1,790 false ones."
- VI: "Bộ phân loại quyết định theo từng khung hình và đôi khi đổi kết quả. Theo vết cho phép quyết định
  một lần cho mỗi vật thể: chiếc xe bị bỏ sót ở một khung hình vẫn được giữ, bụi cây bị nhận nhầm là xe một
  lần thì bị loại. Trên chuỗi 08, việc này lấy lại 1.968 phát hiện đúng và loại bỏ 1.790 phát hiện sai."

**Q12. "Why PointNet for the classifier?"** → slide 15
- EN: "A cluster is an unordered set of points with no grid, so an image network doesn't fit.
  PointNet is built for exactly that: it treats each point the same way and pools them, so the order
  doesn't matter. It's small and fast: about 0.75 ms per cluster."
- VI: "Một cụm là tập hợp điểm không có thứ tự và không có lưới, nên mạng xử lý ảnh không phù hợp. PointNet
  được thiết kế đúng cho trường hợp đó: xử lý mỗi điểm như nhau rồi gộp lại, nên thứ tự không quan trọng.
  Nó nhỏ và nhanh: khoảng 0,75 ms cho mỗi cụm."

### D. Results

**Q13. "Recall 0.73: a quarter of the cars are missed. Is that good enough?"** → slides 31–32
- EN: "For a system with no new labels, it's a measured limit, not a tuning failure. Most misses are
  far, sparse cars, or near cars split into pieces by clustering. The classifier isn't the bottleneck:
  the geometric stage alone already finds 77%. And what it does report is right 91% of the time.
  Against all annotated cars, including those with almost no points, recall is about 0.55; the 0.73
  counts cars with at least ten points left."
- VI: "Với một hệ thống không cần nhãn mới, đây là một giới hạn đã được đo, không phải do tinh chỉnh kém.
  Phần lớn các xe bị bỏ sót là xe ở xa, thưa điểm, hoặc xe ở gần bị phân cụm tách thành nhiều mảnh. Bộ phân
  loại không phải là điểm nghẽn: riêng bước hình học đã tìm được 77%. Và những gì hệ thống báo cáo thì đúng
  91%. Tính trên tất cả các xe được gán nhãn, kể cả xe gần như không có điểm, độ phủ khoảng 0,55; con số
  0,73 tính trên các xe còn ít nhất mười điểm."
- Trap: give the 0.55 yourself if the denominator comes up (S4 quiz gap).

**Q14. "Why is sequence 01 so poor?"** → slide 51
- EN: "It's the highway: cars are mostly far away and recall collapses with range, about 0.2 beyond 40
  metres. The front end was also tuned on dense city scenes. It fails safe, though: precision there is
  still 0.951."
- VI: "Đó là chuỗi đường cao tốc: xe phần lớn ở xa và độ phủ giảm mạnh theo khoảng cách, khoảng 0,2 khi xa
  hơn 40 mét. Phần đầu của hệ thống cũng được tinh chỉnh trên cảnh đô thị dày đặc. Tuy vậy nó thất bại một
  cách an toàn: độ chính xác ở đó vẫn là 0,951."
- Don't volunteer the tracker-gate speculation (S4 trap #8).

**Q15. "How do you know the completed shape is correct?"** → slides 36, 37, 39
- EN: "Exactly correct, we can't know on real data: nobody has seen the hidden side. So we claim
  something we can test: an independent judge accepts completed cars 79% of the time against 25% for
  raw scans, nearly as often as dense real scans of parked cars. On synthetic cars, where the true shape
  exists, the network is within 0.16 m Chamfer distance."
- VI: "Chính xác tuyệt đối thì không thể biết trên dữ liệu thật: chưa ai thấy phía bị che. Vì vậy chúng
  tôi khẳng định điều có thể kiểm chứng: một bộ đánh giá độc lập chấp nhận 79% xe đã hoàn thiện so với 25%
  bản quét gốc, gần bằng các bản quét thật dày đặc của xe đỗ. Trên xe tổng hợp, nơi có hình dạng thật,
  mạng sai lệch trong khoảng 0,16 m theo khoảng cách Chamfer."

**Q16. "Does it work at night or in rain?"**
- EN: "LiDAR measures with its own laser, so darkness is no problem. Rain and fog are a different
  story: they add spurious points and weaken returns. KITTI was recorded in good weather, so we haven't
  tested that, and I wouldn't claim it."
- VI: "LiDAR đo bằng tia laser của chính nó, nên trời tối không phải vấn đề. Mưa và sương mù thì khác: chúng
  tạo ra các điểm nhiễu và làm yếu tín hiệu phản hồi. KITTI được thu trong điều kiện thời tiết tốt, nên
  chúng tôi chưa kiểm tra trường hợp đó, và tôi sẽ không khẳng định điều đó."

**Q17. "What is your main contribution?"** → slide 8, 50
- EN: "Two things. A modular framework that detects and completes occluded cars on real LiDAR, where
  every stage can be inspected. And doing it with zero additional manual annotation: the classifier
  learns from labels mined automatically, and the completion network from synthetic shapes. On top of
  that, the inspection itself produced findings, like tracing the recall limit to clustering."
- VI: "Có hai điểm. Một khung hệ thống dạng mô-đun phát hiện và hoàn thiện các xe bị che khuất trên dữ
  liệu LiDAR thật, với mọi bước đều kiểm tra được. Và làm được điều đó mà không cần thêm nhãn thủ công: bộ
  phân loại học từ nhãn khai thác tự động, còn mạng hoàn thiện học từ hình dạng tổng hợp. Thêm vào đó,
  chính việc kiểm tra từng bước đã cho ra các phát hiện, như truy ra giới hạn độ phủ nằm ở bước phân cụm."
- Trap: objective 2 (evaluation methodology) has no matching contribution; if asked, "the method is
  Chen et al.'s; our part is applying it with an independent judge and a pass mark fixed in advance"
  (project_state weak spots).

**Q18. "What would you do with more time?"** → slide 52
- EN: "Three things, each from a finding: a faster HDBSCAN implementation for real time; fine-tuning
  completion on real data; and testing whether completed cars give better boxes for detection."
- VI: "Ba việc, mỗi việc xuất phát từ một kết quả: cài đặt HDBSCAN nhanh hơn để chạy thời gian thực; tinh
  chỉnh mạng hoàn thiện trên dữ liệu thật; và kiểm chứng xem xe đã hoàn thiện có cho hộp bao tốt hơn khi
  phát hiện hay không."
- Trap: "fine-tuning on real data" is future work, never "tried" (Part 2).
