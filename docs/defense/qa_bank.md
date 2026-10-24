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
| H3 | 31, 3.3.3 Recall | "Six repairs for the splitting failed; one change worked." | "What did you try?" | **56** |
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
  nằm ở mạng, mà ở hệ toạ độ đầu vào. Đó cũng là lý do tinh chỉnh trên các bản quét mô phỏng chân thực hơn không giúp được gì."
- Follow-up "why not just train it on rotated cars?": "We could, but then the network has to learn
  orientation as well as shape from the same data. Putting the scan in a fixed frame is free and exact,
  and it worked."
- Follow-up "how did you find it?" (2026-10-24): own it; it was our integration mistake (#26), not
  something the modular design detected. Never cite it as a modularity win (Q19).
  - EN: "That one was our own mistake. When we connected the network to real scans, we normalised
    them differently from how it was trained. Once we matched the training frame exactly, the same
    network produced cars. The lesson is to check that training and inference conventions match
    before integrating anything, and we now do that as a rule."
  - VI: "Đó là lỗi của chính chúng em. Khi nối mạng với các bản quét thật, chúng em chuẩn hoá đầu vào
    khác với cách mạng được huấn luyện. Khi đưa về đúng hệ toạ độ lúc huấn luyện, cùng một mạng đã cho
    ra hình chiếc xe. Bài học là phải kiểm tra quy ước giữa huấn luyện và suy luận khớp nhau trước khi
    tích hợp, và giờ chúng em làm việc đó thành quy tắc."

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
  fell. What did work was clustering on a 10-centimetre grid instead of 5: recall 0.710 to 0.732.
  It closes the small gaps inside a car before it closes the gaps between two cars."
- VI: "Sáu cách: tăng kích thước cụm tối thiểu, gộp các mảnh ở gần nhau, tham số thay đổi theo khoảng
  cách, phân cụm nhìn từ trên xuống, chồng nhiều khung hình, và hạ ngưỡng kích thước. Mỗi cách thất bại
  vì một lý do rõ ràng. Tăng kích thước tối thiểu giúp ở chuỗi 00 nhưng làm kém đi ở chuỗi 08, vì nó cũng
  xoá mất các xe nhỏ ở xa. Gộp các mảnh gần nhau cũng gộp xe với tường và cột, nên độ chính xác giảm. Thay
  đổi hiệu quả là phân cụm trên lưới 10 cm thay vì 5 cm: độ phủ tăng từ 0,710 lên 0,732. Nó lấp
  những khe nhỏ bên trong một chiếc xe trước khi lấp khe giữa hai chiếc xe."
- Follow-up "slide 56 shows seq 08 F1 0.829, above your 0.812 headline?" (2026-10-24):
  - EN: "That table is an earlier version of the pipeline: the old 5-centimetre grid and a different
    classifier checkpoint. It compares the repairs against each other, row by row; it isn't comparable
    with the headline, which is the final pipeline on the full sequence."
  - VI: "Bảng đó là một phiên bản cũ hơn của hệ thống: lưới 5 cm cũ và một checkpoint bộ phân loại khác.
    Bảng dùng để so sánh các cách sửa với nhau, theo từng hàng; không so sánh được với kết quả chính, vốn
    là hệ thống cuối cùng trên toàn bộ chuỗi."
  - Thesis caption says "pre-promotion baseline" (ch4 tab:strategies). Inferred, not verified: the seq-08
    row (0.956 / 0.731 / 0.829) equals the 100-frame pretrained-checkpoint run in findings.md #25
    exactly, so it is likely a 100-frame number. Don't volunteer "100 frames"; if asked, "an early,
    shorter evaluation run".
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
  counts cars with at least ten points after preprocessing."
- VI: "Với một hệ thống không cần nhãn mới, đây là một giới hạn đã được đo, không phải do tinh chỉnh kém.
  Phần lớn các xe bị bỏ sót là xe ở xa, thưa điểm, hoặc xe ở gần bị phân cụm tách thành nhiều mảnh. Bộ phân
  loại không phải là điểm nghẽn: riêng bước hình học đã tìm được 77%. Và những gì hệ thống báo cáo thì đúng
  91%. Tính trên tất cả các xe được gán nhãn, kể cả xe gần như không có điểm, độ phủ khoảng 0,55; con số
  0,73 tính trên các xe còn ít nhất mười điểm sau tiền xử lý."
- Trap: give the 0.55 yourself if the denominator comes up (S4 quiz gap).
- Follow-up "ten points raw or voxelized?" (2026-10-25): **after preprocessing** (z-filter, 5 cm voxels,
  denoising, ground removal; `src/evaluate.py` `m.sum() >= 10` on the object cloud; ch4
  §sec:eval-targets). Never say "ten points in the scan".
  - EN: "Points after preprocessing: the 5-centimetre voxels, denoising and ground removal, the same
    cloud the clustering sees. A car with fewer than ten points there can't be clustered by any
    method. Some of those cars did have more points in the raw scan, so we also report recall against
    every annotated car: about 0.55."
  - VI: "Là các điểm sau tiền xử lý: voxel 5 cm, khử nhiễu và loại mặt đất, đúng đám mây mà bước phân
    cụm nhìn thấy. Xe còn dưới mười điểm ở đó thì không phương pháp phân cụm nào tách được. Một số xe
    trong đó có nhiều điểm hơn trong bản quét gốc, nên chúng em cũng báo cáo độ phủ trên tất cả xe được
    gán nhãn: khoảng 0,55."
- If pressed "so preprocessing loses cars you don't count?": thesis stride-20 seq 08: 2,332 annotated
  → 2,033 with ≥ 10 raw points → 1,737 eligible; 14.6% of the raw-≥10 cars drop out in preprocessing.
  - EN: "Yes, some: about 15% of cars with ten or more raw points drop below ten after
    preprocessing. That's a real cost of the front end, and it's why we put 0.55 next to 0.73
    instead of reporting only the higher number."
  - VI: "Đúng, có một phần: khoảng 15% số xe có từ mười điểm trở lên trong bản quét gốc bị giảm xuống
    dưới mười sau tiền xử lý. Đó là cái giá thật của phần đầu hệ thống, và vì thế chúng em đặt 0,55 cạnh
    0,73 thay vì chỉ báo cáo con số cao hơn."

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

### E. The modular design (2026-10-24)

**Q19. "You built it modular to see where it fails. What did that actually reveal?"** → slides 31, 33
(29, 44, 41)
- EN: "Two concrete things, plus one practical one. First, recall is limited by clustering, not by
  the classifier. After the geometric stage alone, 77% of cars are already found, more than the full
  pipeline keeps. And a third of cars are split into pieces by clustering, while fewer than 1% are
  merged. So we knew improving the classifier could not raise recall, and we fixed clustering instead:
  a coarser grid took recall from 0.710 to 0.732. Second, each stage's job can be measured: the
  classifier cuts false positives from about 109 thousand to 4,300, and the track filter to 2,500. And
  per-stage timing shows clustering takes 59% of the runtime, so that's where real-time work has to
  start."
- VI: "Có hai phát hiện cụ thể và một ý thực tế. Thứ nhất, độ phủ bị giới hạn bởi bước phân cụm, không
  phải bộ phân loại. Chỉ sau bước hình học, 77% số xe đã được tìm thấy, nhiều hơn cả hệ thống hoàn chỉnh
  giữ lại. Và một phần ba số xe bị phân cụm tách thành nhiều mảnh, trong khi chưa đến 1% bị gộp. Vì vậy
  chúng em biết cải thiện bộ phân loại không thể tăng độ phủ, và sửa ở bước phân cụm: lưới thô hơn giúp
  độ phủ tăng từ 0,710 lên 0,732. Thứ hai, vai trò của từng bước đo được: bộ phân loại giảm dương tính
  giả từ khoảng 109 nghìn xuống 4.300, bộ lọc theo vết xuống 2.500. Và đo thời gian từng bước cho thấy
  phân cụm chiếm 59% thời gian chạy, nên muốn chạy thời gian thực thì phải bắt đầu từ đó."
- Trap: **never cite the PCN normalisation fix here**; it was our own bug (H1 follow-up). Don't claim
  per-car blame ("we know where each missed car was lost"); the thesis measures rates and per-stage
  recall. Numbers: tab:ablation-stage (FP 108,831 → 4,303 → 2,513), tab:splitrate (33.7% split, 0.8%
  merged), tab:runtime (370.4 ms, 59.4%).
- Follow-up "end-to-end networks can be ablated too":
  - EN: "Ablation tells you which component matters. Here every stage outputs points we can compare
    directly with the labels, so we can measure how many cars are already split right after
    clustering, and how much recall is left after each stage."
  - VI: "Ablation cho biết thành phần nào quan trọng. Ở đây mỗi bước cho ra các điểm có thể so trực tiếp
    với nhãn, nên chúng em đo được bao nhiêu xe đã bị tách ngay sau bước phân cụm, và độ phủ còn lại sau
    từng bước."
- Follow-up "you found the splitting; why not fix it fully?" → slide 56:
  - EN: "Six repairs failed for the same reason: density clustering has no notion of an object. No
    distance or size rule can tell two pieces of one car from two cars parked side by side. That's a
    limit of the method and our future work."
  - VI: "Sáu cách sửa đều thất bại vì cùng một lý do: phân cụm theo mật độ không có khái niệm vật thể.
    Không quy tắc khoảng cách hay kích thước nào phân biệt được hai mảnh của một xe với hai xe đỗ sát
    nhau. Đó là giới hạn của phương pháp và là hướng phát triển tiếp."
- Follow-up "what's the downside of modular?" → slide 33 (say it before they push):
  - EN: "Errors carry forward. The classifier and tracker can't create detections, so recall can never
    exceed the 0.773 of the geometric stage."
  - VI: "Lỗi dồn từ bước trước sang bước sau. Bộ phân loại và bộ theo vết không tạo thêm được phát hiện,
    nên độ phủ không bao giờ vượt quá 0,773 của bước hình học."

**Q20. "If the geometric stage is the limit, was it the right call to put it first?"** → slides 29, 33
(15, 41)
- EN: "For the question this thesis asks, yes, and we measured the price. Three reasons. It needs no
  labels at all, which is the point of the thesis. It turns over a hundred thousand points into a few
  dozen candidate clusters per scan, so the classifier stays small and fast. And those clusters are what
  we mine the classifier's training labels from: without that first step there is nothing to label
  automatically. The price is that recall can't go above what clustering finds, 0.773. A learned
  first step would probably find more cars, but it needs a label on every point, which is exactly
  what we set out to avoid. And the limit isn't fixed: changing the clustering grid alone raised
  recall from 0.710 to 0.732."
- VI: "Với câu hỏi mà luận văn đặt ra thì đúng, và chúng em đã đo cái giá của nó. Có ba lý do. Bước này
  hoàn toàn không cần nhãn, đó chính là mục tiêu của luận văn. Nó biến hơn một trăm nghìn điểm thành
  vài chục cụm ứng viên mỗi lần quét, nên bộ phân loại nhỏ và nhanh. Và chính các cụm đó là nguồn để
  khai thác nhãn huấn luyện cho bộ phân loại: không có bước đầu tiên này thì không có gì để gán nhãn tự
  động. Cái giá là độ phủ không thể cao hơn những gì bước phân cụm tìm được, 0,773. Một bước đầu dùng học
  máy có lẽ sẽ tìm được nhiều xe hơn, nhưng cần nhãn cho từng điểm, đúng thứ mà chúng em muốn tránh. Và
  giới hạn này không cố định: chỉ riêng việc đổi lưới phân cụm đã nâng độ phủ từ 0,710 lên 0,732."
- Count check: seq 08 has ~222 clusters per frame after clustering, ~31 after the geometric filter
  (903,901 / 127,654 over 4,071 frames); the runtime table's ~45 is the classifier's per-frame load.
  Say "a few dozen".
- Precision of the premise: the limit is **clustering** (splitting), not "geometry" as a whole; the
  geometric filter's job is to keep cars, and geometric-only recall 0.773 is the highest of any stage.
- Trap: "a learned first step would probably find more" is an expectation, not measured; keep
  "probably". Don't call geometry-first optimal for accuracy. Don't bring up the RANSAC wall frames
  (project_state weak spots).
- Follow-up "you already use SemanticKITTI labels; why not train a segmentation network on them?":
  - EN: "That's a valid alternative, but then it's the fully supervised setting we compare against, not
    the one we study. Our design only needs one car-or-not label per cluster, and we never train on
    point-level labels. The question here is how far you get without them."
  - VI: "Đó là một hướng hợp lý, nhưng khi đó nó trở thành bài toán học có giám sát đầy đủ mà chúng em
    dùng để so sánh, không phải bài toán chúng em nghiên cứu. Thiết kế của chúng em chỉ cần một nhãn xe
    hay không phải xe cho mỗi cụm, và không bao giờ huấn luyện trên nhãn từng điểm. Câu hỏi ở đây là đi
    được bao xa khi không có chúng."
  - Trap: don't claim cluster labels "could come from cheap clicking"; not tested.
