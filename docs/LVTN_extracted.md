# LVTN.pptx — Full Extraction (Reference Baseline)

> Source: `docs/LVTN.pptx` (63 slides, 15.6 MB). **This is a peer's** master's-thesis
> defense deck, NOT our LiDAR project. Kept as a **structure + content baseline** for
> when we build our own thesis defense deck.
>
> **Title:** *Event-Driven Multimodal Traffic Scenario Retrieval Using Spatio-Temporal
> Symbolic Graphs* — Trinh Thi Mai Huong (ID 23MSE13176), supervisor Dr. Doan Nhat Quang.
>
> Extraction method: python-pptx (text), `olefile`+`openpyxl` (26 embedded Excel tables →
> exact numbers), and direct image viewing (figures/schemas). Table numbers below are the
> *authoritative* embedded-Excel values; slide-rendered tables match them.

---

## 1. Deck structure (the reusable skeleton)

- **I. Introduction** (s3–8): background → current approaches → literature review →
  research gap → objectives & contributions.
- **II. Proposed Methodology** (s9–18): system architecture → spatio-temporal graph
  construction pipeline → detailed spatial/temporal graph examples → traffic ontology →
  rule-based event engine → user input processing → **IA-HGA alignment** → temporal
  matching pipeline.
- **III. Experiment Evaluation** (s19–51): dataset + human-in-the-loop GT → object/relation
  stats → text/JSON structures → experiment setup (baselines, metrics, splits) → results
  (retrieval + classification, per-case "baselines vs ours" walkthroughs, analysis) →
  runtime → ablation → parameter sensitivity.
- **IV. Conclusion & Future Work** (s52–56): summary → limitations → discussion/future work
  → thank-you.
- **Appendix / backup** (s57–63): fully-automated end-to-end results, "why is it low"
  VLM-SGG bottleneck analysis, triplet-parsing quality, full result tables, and a
  Vietnamese defense Q&A script (IA-HGA vs Naive Hungarian).

**Patterns worth copying:** (a) "CASE N — BASELINES VS. OURS" walkthrough slides narrating
why each baseline fails on a concrete example; (b) per-method "RESULT ANALYSIS" slides with
Strengths/Limitations; (c) appendix backup slides with full tables + failure deep-dives.

---

## 2. Introduction content

**Research gap (s7):** semantic representation (visual-only, weak interaction modeling),
explainability (black-box hazard outputs), temporal understanding (frame-level only),
flexible multimodal retrieval (limited text/image/video querying).

**Objectives (s8):** (1) multimodal traffic query [text/image/video]; (2) semantic traffic
understanding [spatial + behavioral relations as graphs]; (3) dynamic event understanding
[spatio-temporal graphs]; (4) explainable retrieval [behaviorally-similar scenarios + evidence].

**Contributions (s8):** (1) multimodal query framework; (2) spatio-temporal dynamic graph
representation (extends static scene graphs, models object lifecycles + events);
(3) **Semantic-Guided Interaction-Centric Hungarian Graph Alignment (IA-HGA)**;
(4) retrieval-based hazard reasoning (ranked weighted voting, explainable).

> ⚠️ Acronym inconsistency in the deck: IA-HGA is expanded as both "…**Hungarian** Graph
> Alignment" (title slides) and "Interaction-Aware **Hierarchical** Graph Alignment"
> (Q&A script). Pick one in ours.

**Literature review table (s6, from oleObject1):**

| Approach | Representative Study | Venue | Strengths | Remaining Challenges |
|---|---|---|---|---|
| End-to-End Deep Learning | Yurtsever 2019; Fang 2022 (DADA) | IEEE ITSC; IEEE T-ITS | End-to-end; spatial+temporal visual features | Low explainability; pixel-based; weak interaction semantics |
| Spatio-Temporal Graph Learning | Bao 2020; Zhou 2024 (HKTSG) | ACM MM; IEEE T-IV | Models ST interactions; object relations; graph knowledge | Data-driven; limited symbolic reasoning; weak retrieval |
| LLM/VLM Semantic Graph | Tian 2025 (Query by Example); Song 2025 (Plan Then Retrieve) | Sensors; ACM WWW | Structured semantics; multimodal; logical reasoning | Mostly static graphs; weak temporal; no actor/target in undirected graphs |

---

## 3. Methodology content

### Ontology (node types, s14 / oleObject3)
`ego_veh` (ego vehicle), `road`, `tl` (traffic light), `veh` (car/van/truck), `cyc`
(cyclist), `ped` (pedestrian), `obs` (obstacle: animal/unidentified), `bar` (barrier).

### Relations (s23 / oleObject2)
`very_near, near, visible, getting_close_to, approaching, direct_front, on_the_left,
on_the_right, locate_here, successor, driving_on, parking_on/waiting_on, following,
overtaking` (+ from charts: `direct_rear, on_same_lane, turn_left, turn_right, go_straight,
on_the_opposite, crossing, yielding_to, waiting_at_crossroad, walking_along`).

### Attribute categories (s24 / oleObject4)
weather (Sunny/Rain/Snow/Fog/Sandstorm/Haze/Cloudy), lighting (Day/Night/Dusk/Dawn),
road_type (Highway/Urban/Rural/Mountain/Intersection/Bridge/Tunnel), traffic
(Heavy/Light), environment (Urban/Suburbs/Countryside/Intersection zone),
unusual_driving_behavior (Lane change/Highway merge/Exit/Lose control/Crash/Brake/Driving fast).

### Event types (s15 / oleObject5) — spatial-region transition (prev → curr)
- `vehicle_crossing`: left/right/approaching/None → front
- `vehicle_overtake_ego`: rear → front (overtakes) / front → left|right (dodge) / → None; passing_by/overtaking relation
- `pedestrian_crossing`: left/right → front
- `vehicle_lane_change`: left ↔ right
- `object_appeared_front`: None → front
- `exited_lane`: object leaves frame while previously `is_in` lane
- `disappeared`: object leaves frame entirely beyond N frames

### Spatial-graph JSON schema (s25, image106)
```json
{ "graph": {
  "nodes": [{"name":"ROAD","type":"road"}, {"name":"ego_veh","type":"ego_veh"},
            {"name":"veh_1","type":"vehicle"}, {"name":"veh_2","type":"vehicle"}],
  "attributes": [{"belong_to":"veh_1","value":"hazard"},
                 {"belong_to":"road","value":"Sunny"},
                 {"belong_to":"road","value":"Light_Traffic"}],
  "edges": [{"object_1":"ego_veh","object_2":"ROAD","type":"driving_on"},
            {"object_1":"veh_1","object_2":"ego_veh","type":"to_the_left"},
            {"object_1":"veh_1","object_2":"ego_veh","type":"very_near"}] } }
```

### Temporal-graph JSON schema (s26, image107) — adds frames + events
```json
{ "global_attributes": ["Sunny","Light_Traffic"],
  "graph": { "nodes": [...],
    "edges": [{"object_1":"ego_veh","object_2":"ROAD","type":"driving_on",
               "start_frame":82,"end_frame":82}, ...] },
  "events": [
    {"event_id":"vehicle_disappeared_veh_2_82","event_type":"vehicle_disappeared",
     "actor":"veh_2","target":"ROAD","trigger_frame":82,"reasoning":"veh_2 disappear road"},
    {"event_id":"vehicle_crossing_veh_1_82","event_type":"vehicle_crossing",
     "actor":"veh_1","target":"ego_veh","trigger_frame":82,"reasoning":"veh_1 crossing ego_veh"} ] }
```

### IA-HGA alignment (s17)
Pipeline: **2-hop ego-veh subgraph extraction** (prune road/static nodes) → **Hungarian
node assignment** → **fuzzy relation-edge matching strictly within ontology** → score & rank.
- Cost matrix example (image79): Query nodes × scene candidates, best match highlighted
  (veh_1→veh_a=1.0; ped_1→ped_d(near)=0.9).
- Retrieval pipeline (s18): query encoded to embeddings → vector DB retrieves candidate
  graphs → IA-HGA only on candidates (not whole KB) → event matching + subgraph/attribute matching.

### Method comparison for ablation (s47 / oleObject19-20)
- **SSI (Strict Subgraph Isomorphism):** binary 0/1 match; `ped_1—near→ego` vs
  `ped_a—very_near`/`ped_b—visible` ⇒ ∅ ⇒ 0.0.
- **ISRA (Independent Semantic Relation Alignment):** semantic coverage ratio, no 1-to-1
  entity mapping, no topo check; matches both (0.90 & 0.80 over threshold 0.6).
- **IA-HGA:** Hungarian node mapping M* then topo-edge validation per M*; picks 0.90 > 0.80.

---

## 4. Experiment setup

**Dataset (s20):** Road Hazard Stimuli — original 750 clips (434 hazard / 316 safe); this
thesis uses **300 videos (200 hazard / 100 safe)**. Categories (image97): Animal, Cyclist,
Obstacle, Pedestrian, Vehicle, Other.

**Object/label totals (s22):** 9,575 frames / 300 videos; Hazard 8,275 / 200 videos;
Safe 1,500 / 100 videos.

**Baselines (s28 / oleObject9):**
- **SBERT** — pure semantic dense-vector retrieval (cosine), text queries.
- **RSG-LLM** — static scene-graph retrieval via exact subgraph isomorphism (VF2), LLM/VLM graphs.
- **ResNet50 + LSTM** — end-to-end black-box CNN+LSTM (roadscene2vec infra), raw video.
- **UString (Bao et al.)** — ST-GNN (GCN+RNN) with Bayesian uncertainty, bbox features + adjacency.

**Metrics (s29):** retrieval = Hits@K(1,6,10), Recall@K, MRR, F1@K; classification =
Accuracy, Precision, Recall, F1.

**Data split (s30):**
- Retrieval (oleObject12): Spatial — query 300 frames / KB 9,275 frames; Temporal —
  query 4,652 / KB 4,923 (interleaved sampling stride=1).
- Classification (oleObject13): Spatial — 100 test frames (70 haz/30 safe) / KB 6,396,
  strict isolation; Temporal — 100 graphs (70/30) / KB 200 graphs, video-level isolation.

---

## 5. RESULTS (exact numbers from embedded Excel)

### 5.1 Retrieval — Spatial (oleObject14)
| Method | H@1 | R@1 | F1@1 | H@6 | R@6 | F1@6 | H@10 | R@10 | F1@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| SBERT | .718 | .718 | .718 | .842 | .842 | .241 | .866 | .866 | .157 | .765 |
| RSG-LLM* | .535 | .535 | .535 | .690 | .690 | .197 | .735 | .735 | .134 | .596 |
| **Ours** | **.750** | .750 | .750 | **.927** | .927 | .265 | **.943** | .943 | .172 | **.819** |

### 5.2 Retrieval — Temporal (oleObject15)
| Method | H@1 | H@6 | H@10 | MRR |
|---|---|---|---|---|
| SBERT | .377 | .597 | .663 | .462 |
| RSG-LLM* | -- | -- | -- | -- |
| **Ours** | **.717** | **.887** | **.913** | **.783** |

### 5.3 Classification (oleObject16)
| Task | Model | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|
| Static | ResNet-50+LSTM | .48 | .696 | .457 | .552 |
| Static | GNN-UString | .86 | .842 | **.986** | **.908** |
| Static | **OURS** | .81 | **1.000** | .729 | .843 |
| Temporal | ResNet-50+LSTM | .63 | .685 | .871 | .767 |
| Temporal | GNN-UString | .83 | .812 | .986 | .890 |
| Temporal | **OURS** | **.96** | **.971** | **.971** | **.971** |

Notable: Ours has **highest Static Precision (1.000, zero FP)** but lower Static recall;
dominates on Temporal. UString is a strong classification baseline (high recall).

### 5.4 Runtime (oleObject17-18) — Ours is an *online* pipeline, so slowest
- Retrieval: RSG-LLM 21.07s(img)/22.61s(txt); Ours 20.15s(img)/10.52s(txt)/33.18s(video).
- Classification: ResNet50-LSTM 1.18–2.18s; UString 4.07–4.24s; **Ours 23.07s(img)/39.66s(video)**.

### 5.5 Ablation (oleObject21-22) — SSI vs ISRA vs IA-HGA
Spatial: | Method | H@1 | H@6 | H@10 | MRR | Acc | Rec | F1 |
|---|---|---|---|---|---|---|---|
| SSI | .637 | .777 | .823 | .696 | .710 | .580 | .727 |
| ISRA | .680 | .820 | .830 | .733 | .820 | .234 | .680 |
| **IA-HGA** | **.750** | **.927** | **.943** | **.819** | **.862** | **.825** | **.894** |

Temporal: | Method | H@1 | H@6 | H@10 | MRR | Acc | Rec | F1 |
|---|---|---|---|---|---|---|---|
| SSI | .190 | .380 | .433 | .270 | .347 | .020 | .039 |
| ISRA | .647 | .853 | .893 | .729 | .957 | .935 | .966 |
| **IA-HGA** | **.727** | **.893** | **.920** | **.791** | **.967** | **.950** | **.974** |

> The Q&A script (s63) also cites a **Naive Hungarian** baseline that *beats* IA-HGA on
> Spatial retrieval H@1 (.807 vs .750) — framed as an intentional trade-off (sacrifice
> retrieval H@1 for downstream classification robustness). Spatial classification: Naive
> .8567 vs Ours .862. This "we lose on metric X on purpose" argument should be on a visible
> slide, not buried in backup.

### 5.6 Parameter sensitivity
Spatial (oleObject23) — relation vs attribute weight; **S2 (0.65/0.35) selected**:
| Config | w_rel | w_attr | H@1 | H@6 | H@10 | MRR | R@6 | F1@6 |
|---|---|---|---|---|---|---|---|---|
| S1 | .70 | .30 | .64 | .83 | .87 | .716 | .83 | .237 |
| **S2** | **.65** | **.35** | .64 | .85 | .87 | .718 | .85 | .243 |
| S3 | .60 | .40 | .65 | .86 | .87 | .723 | .86 | .246 |
| S4 | .50 | .50 | .65 | .85 | .88 | .724 | .85 | .243 |

Temporal (oleObject24) — event/relation/attribute; **TB (0.50/0.30/0.20) selected**:
| Config | Event | Rel | Attr | H@1 | H@6 | H@10 | MRR |
|---|---|---|---|---|---|---|---|
| TA | .45 | .35 | .20 | .717 | .887 | .913 | .783 |
| **TB** | **.50** | **.30** | **.20** | .727 | .893 | .920 | .791 |
| TC | .40 | .45 | .15 | .680 | .840 | .883 | .745 |
| TD | .35 | .35 | .30 | .730 | .903 | .937 | .797 |
(TD scores slightly higher but over-weights context; TB chosen for interaction-centric reasoning.)

---

## 6. Appendix / limitations (the honest part — worth emulating)

**Fully-automated end-to-end (s57 / oleObject25):** no human correction (YOLOv8 → track →
VLM graph → HGA → voting). Performance collapses:
| Metric | Temporal (50 vid) | Spatial (50 img) | Overall |
|---|---|---|---|
| Hits@1 | .1346 | .0263 | .0889 |
| MRR | .1682 | .0483 | .1176 |
| Class. Accuracy | .4615 | .500 | .4778 |
| F1 | .5758 | .6122 | .5913 |

**VLM triplet-parsing quality (s59 / oleObject26):** strict (Subj,Pred,Obj) match.
| Metric | Temporal | Spatial | Overall |
|---|---|---|---|
| Triplet Precision | .5921 | .3421 | .4866 |
| Triplet Recall | .1966 | .1136 | .1616 |
| Triplet F1 | .2811 | .1624 | .2426 |

**Diagnosis (s58, s60):** performance is bottlenecked by the **VLM visual parser, not the
retrieval matcher** — semantic-structural gap, hallucinated relations, 2D depth/left-right
ambiguity, error cascading in the neuro-symbolic pipeline. Temporal beats Spatial because
motion gives a precision anchor. Relation-frequency chart (s61, image185) shows GT-vs-VLM
edge counts per relation — VLM notably under-generates `successor` (2352→492),
`turn_left` (498→72), `on_the_opposite` (474→138); over-generates `direct_front`,
`visible`, `very_near`.

**Stated limitations (s54):** VLM dependence (hallucination; RSG-LLM cites 76.95% structural
failure zero-shot, <1% relation recall unconstrained); high online latency (VLM dominates,
not graph matching); no true temporal graph alignment yet (timestamps used only for
explainability).

**Future work (s55):** temporal-aware graph alignment jointly optimizing spatial+temporal
similarity; model event ordering, trajectory evolution, causal interaction sequences.

---

## 7. Fix-list flagged for the peer's deck (not ours)
1. **s63 leaked chat transcript** — a pasted LLM Q&A (Vietnamese, with "12:39"/"12:41"
   timestamps and garbled inline-LaTeX) is sitting in a slide. Remove / move to private notes.
2. **Acronym**: "Hungarian" vs "Hierarchical" Graph Alignment — unify.
3. **Naive-Hungarian trade-off** must be on a visible slide (first question otherwise).
4. **Typos**: "Groud Truth" (s21 ×2), "Reasone and Advice" (s24), "ATRIBUTE" (s18),
   "sceranio" (s55).

---

## 8. Extraction artifacts (this session)
- Unpacked deck: `…/scratchpad/lvtn_extract/ppt/` (185 media files, 26 oleObjects).
- Media map (which image on which slide) + all 26 Excel tables were dumped to terminal.
- Media inventory: 185 files = content figures + png/svg duplicate pairs + decorative icons
  (e.g. image36/38 = node-graph clip-art; image176 = decorative circle) + individual
  walkthrough scene-photo thumbnails + table-render PNGs (data == embedded Excel).
