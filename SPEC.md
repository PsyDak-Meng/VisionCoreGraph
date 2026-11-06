# VisionGraph-Core — Technical Design Specification

**Version:** 2.1.0 (post-design-review)
**Status:** Architecture locked for v0. Implementation-ready.
**Scope:** Modular, SDK-first Python library converting video into a queryable temporal knowledge graph.

---

## How to read this document

| Section | Read if you want |
|---|---|
| §1 | Why this exists and what it refuses to do |
| §2 | The 15 architectural decisions, each with its runner-up and revisit trigger |
| §3 | End-to-end data flow, diagrammed |
| §4 | Per-module algorithms, interfaces, failure modes |
| §5 | The graph data model |
| §6 | The query IR and its Cypher lowering |
| §7 | Cache and resumability semantics |
| §8 | v0 scope and milestone ordering |
| §9 | Evaluation plan and metric definitions |
| §10 | Repo layout, config, packaging |
| §11 | Risks and open items |

Anything marked **⚠ VERIFY** is an assumption that blocks a design and must be tested before that code is written.

---

## 1. Problem and Position

### 1.1 What breaks with naive video retrieval

The default approach — embed every frame (or every N-th frame) into a vector index and retrieve by cosine similarity — fails on three specific classes of question:

| Failure | Example question | Why embeddings fail |
|---|---|---|
| **Context collapse** | "Who was holding the red bag?" | A frame embedding is a single point. It cannot separate *which* of five people in the frame is holding *which* of three bags. Relational structure is averaged away. |
| **Temporal blindness** | "Did he drop the bag *before* leaving?" | Cosine similarity has no ordering operator. Two frames retrieved by similarity carry no reliable statement about their causal order relative to a third event. |
| **Identity fragmentation** | "How long was she in the building?" | The same person exiting and re-entering produces unrelated embeddings. There is no entity to accumulate duration against. |

### 1.2 The position

Treat video as a stream of **semantic deltas** over a **typed temporal graph**. Structure is extracted once at ingest, with types enforced at the boundary, and questions are answered by *deterministic graph queries* rather than by approximate retrieval.

The corollary — and it is a real cost — is that this system can only answer questions expressible in its declared ontology. An open-ended "describe the vibe of this footage" is out of scope by construction. That tradeoff is the entire bet.

### 1.3 Non-goals for v0

- Live/streaming ingestion (see D5).
- Open-vocabulary entity discovery (see D4).
- Video generation or editing (the MCP surface enables it downstream; the SDK does not do it).
- Audio. Everything here is visual-only.

---

## 2. Decision Log with Tradeoff Backup

Fifteen decisions. Each lists the chosen option, the closest rejected alternative, why it lost, and the condition under which the decision should be reopened.

---

### D1 — Audience: OSS SDK for external developers

**Chosen.** A pip-installable library with a stable public API, pluggable backends, and a documented extension path.

| | |
|---|---|
| **Runner-up** | Internal tool for one concrete corpus |
| **Why not** | Would let us hardcode the VLM, store, and schema and reach a working system perhaps 10× faster — but every module boundary would be arbitrary and the eventual generalization would be a rewrite, not a refactor. |
| **Cost accepted** | Interface stability, docs, two implementations of every backend protocol, and a real DX budget. |
| **Revisit if** | After 6 months there are no external users; then collapse abstractions aggressively. |

**Downstream force:** every backend protocol must have **two** real implementations before it ships. A protocol with one implementation is a guess.

---

### D2 — Hosted VLM is the default; local inference is opt-in

**Chosen.** `pip install visiongraph` + an API key runs the quickstart on any laptop. `pip install visiongraph[local]` adds the SGLang backend.

| | |
|---|---|
| **Runner-up** | GPU-required, SGLang-only (the v2.0.0 position) |
| **Why not** | Caps the audience to people with an A100-class machine and kills drive-by contribution. An OSS SDK whose README cannot be executed by its reader is a whitepaper. |
| **Also rejected** | Tiny local VLM (SmolVLM / Qwen2-VL-2B on CPU/MPS) — no API key needed, but extraction quality is poor enough that the resulting graph is mostly noise. A bad first impression is worse than a required key. |
| **Also rejected** | Ship a pre-extracted demo graph — great demo, but the user cannot run their own video, which is the product. |
| **Revisit if** | Small vision models close the relation-extraction gap; then a CPU default becomes strictly better. |

---

### D3 — Filter: MV coarse gate → embedding delta, with first-class ego-motion behind auto-detection

**Chosen.** Phase 1 reads compressed-domain motion vectors as a near-free gate. A global-motion probe decides whether ego-motion compensation runs. Phase 2 dedupes survivors by embedding distance.

| | |
|---|---|
| **Runner-up** | Static-camera-only; defer ego-motion entirely |
| **Why not** | Chosen against because the target modality is **both** static and moving cameras. On a moving camera an uncompensated MV gate keeps essentially every frame, so the filter does no work at all — the component would be decorative. |
| **Also rejected** | Embedding-delta only, no compressed domain — dead simple and container-agnostic, but burns full decode on every frame, which is the cost the compressed-domain trick exists to avoid. |
| **Also rejected** | PySceneDetect — excellent on edited video, useless on fixed CCTV where an 8-hour recording is one shot. |
| **Honest cost** | Ego-motion is ~1–2 days of code, but its threshold is **untunable without an eval set**, and its failure mode is silent: a dropped frame becomes a missing edge that no error surfaces. |
| **Revisit if** | HEVC MV export turns out unavailable (⚠ VERIFY, §11.1); then Phase 1 degrades to decode-and-embed for a large share of real inputs and the compressed-domain premise weakens. |

---

### D4 — Detector and tracker own boxes and `local_id`; the VLM emits relations only

**Chosen.** YOLO/RT-DETR produces boxes; ByteTrack assigns within-tracklet ids; the VLM receives an annotated frame and returns only relationships between given ids.

| | |
|---|---|
| **Runner-up** | VLM does detection, tracking, and relations in one call (the v2.0.0 position) |
| **Why not** | Three compounding problems. (a) VLM box coordinates are typically off by tens of pixels. (b) `local_id` assigned by a model that sees one frame at a time is hallucination-prone across frames. (c) Critically, `GlobalReID` **crops those boxes** to feed OSNet — so a misaligned box yields a garbage appearance embedding, which silently corrupts the canonical registry. The error propagates from the least reliable component into the least recoverable one. |
| **Also rejected** | Open-vocab detector (GroundingDINO / OWLv2) + VLM relations — keeps open vocabulary with grounded boxes, genuinely better long-term, but adds a second heavy model to v0. Deferred, not dismissed. |
| **Also rejected** | VLM boxes refined by SAM before Re-ID — salvages the single-model story by adding a model anyway; strictly worse than doing detection properly. |
| **Cost accepted** | A fixed `label_set` is now required. Open-vocabulary users must wait for the GroundingDINO backend. |
| **Revisit if** | Frontier VLM grounding accuracy reaches detector parity on your domain (measure it; don't assume it). |

---

### D5 — Re-ID is offline constrained clustering, not online greedy matching

**Chosen.** After all deltas exist, each tracklet is reduced to one robust embedding and all tracklets are clustered globally under a temporal-exclusion constraint.

| | |
|---|---|
| **Runner-up** | Online greedy match against a growing registry above a similarity threshold (the v2.0.0 position) |
| **Why not** | Three defects. (a) **Order dependence** — the same video processed twice can yield different identities, violating the determinism the whole system is sold on. (b) **Irreversibility** — one bad match permanently fuses two entities, and there is no signal that it happened. (c) **Compounding** — over an 8-hour video, identity error accumulates monotonically. |
| **Also rejected** | Online with a deferred-commit buffer (provisional ids, retroactive splits within a window) — genuinely the best of both, and the hardest to get right; it also requires entity-merge support in the store. Post-v1. |
| **Also rejected** | No global Re-ID; tracklets as nodes with `SAME_AS` edges added later — ships fastest, but is precisely the node explosion the system exists to prevent. |
| **Cost accepted** | **v0 is batch-only. Live streaming is out of scope.** Graph writes are deferred to the end of ingest. |
| **Bonus** | Because `pipeline.fit(video)` is already batch, this costs nothing in API semantics — the spec's own interface was never streaming. |
| **Revisit if** | A streaming use case becomes primary; then implement the deferred-commit buffer plus store-level merge. |

---

### D6 — Events become state intervals via a declarative reducer

**Chosen.** Each durative relation declares which event types open and close it. A deterministic reducer replays the timestamp-ordered event stream to materialize intervals.

| | |
|---|---|
| **Problem being fixed** | A genuine hole in v2.0.0. The Pydantic contract emits `TemporalEdge(source, target, relation, timestamp)` — an **instantaneous event**. The example Cypher queries `[r:POSSESSES]` filtered on `r.valid_from` / `r.valid_until` — a **durative state**. Nothing converted one into the other. `DROPS @ t=12.4` does not tell the graph that `POSSESSES` held from 3.1 to 12.4. |
| **Runner-up** | VLM emits `valid_from` / `valid_until` directly |
| **Why not** | Structurally unsound: the VLM sees one frame. It cannot know when a state *ends*, so it will fabricate the interval. This moves a correctness guarantee into a hallucination. |
| **Also rejected** | Two edge classes with no conversion; coalesce observed per-keyframe states at query time — simpler ingest, but every query pays reconstruction cost, and a gap created by *frame pruning* becomes indistinguishable from a real state change. The filter would silently corrupt query results. |
| **Also rejected** | Sliding-window VLM over N consecutive keyframes — better grounded and worth doing for extraction quality, but intervals spanning windows still need stitching, so the reducer is required regardless. Orthogonal improvement, not an alternative. |
| **Cost accepted** | Every skill schema must author transition rules for its durative relations, not just entity classes. |

---

### D7 — Transition rules: LLM-drafted, human-committed to code

**Chosen.** `visiongraph schema scaffold` has an LLM propose the `opens_on` / `closes_on` / cardinality table from the user's entity and relation list, and writes it to a reviewable Python file.

| | |
|---|---|
| **Why rules can't be skipped** | "Deterministic and queryable" *requires* closure semantics to be declared somewhere. The only alternative is an LLM inferring at query time whether `HANDS_OFF` closes `POSSESSES` — which makes the same question return different answers on different runs. That is exactly what rejecting vector retrieval was supposed to buy. |
| **Authoring burden, honestly** | Most users pick a shipped schema and author nothing. New-domain authors write ~20 lines. Users asking only event-shaped questions author nothing, since instantaneous relations need no rules. |
| **Runner-up** | Hand-authored only |
| **Why not** | Zero magic and it forces ontology thinking, but it is a real onboarding wall for an OSS SDK. |
| **Also rejected** | Convention-based defaults from an English-verb lexicon (`PICKS_UP`/`DROPS`, `ENTERS`/`EXITS`) — lowest friction, but a silently wrong default is worse than a loudly missing rule. |
| **Also rejected** | Rules optional; unruled relations stay event-only and interval queries return nothing — degrades gracefully but the failure is an empty result set, the single worst failure mode in this system. |
| **Key property** | Inference happens **once, at authoring time**. Runtime is fully deterministic. The generated file is diffable, reviewable, and unit-testable. |

---

### D8 — Retrieval is a composable temporal IR compiled to Cypher

**Chosen.** The LLM emits a validated Pydantic IR tree over a small set of domain-independent operators. A tested compiler lowers the IR to Cypher.

| | |
|---|---|
| **Key insight** | The *question types* are domain-independent; only the *vocabulary* is domain-specific. "Was S true at t?", "did A precede B?", "what overlapped X during I?" is a temporal algebra. The schema supplies the type names it operates over. So the thing to build is a compiler, not a per-domain template library. |
| **Runner-up** | Free-form text-to-Cypher with schema in context and a validation retry loop |
| **Why not** | No ceiling and nothing to build, but temporal-interval logic is exactly where generated Cypher is most often *subtly* wrong: off-by-one on `valid_until`, missing `IS NULL` handling on open intervals, inverted precedence. "Validates and runs" ≠ "correct" — a syntactically valid query with wrong temporal logic returns a confident wrong answer with no signal at all. |
| **Also rejected** | Hybrid: IR first, generated-Cypher fallback labeled unverified — pragmatic and honest, and likely the v1 answer. Rejected for v0 because two code paths double the test surface before either is proven, and users do not read confidence labels. |
| **Also rejected** | No NL layer; ship the IR as a typed Python/MCP API — smallest surface, arguably the most honest SDK answer, and the IR is valuable standalone. But it deletes the one-liner from the README. The IR *is* exposed publicly; the NL layer just sits on top of it. |
| **Cost accepted** | The algebra's boundary is a hard expressiveness ceiling. Questions outside it must fail loudly, not degrade. |

---

### D9 — Ground truth: public VidSGG **and** an owned end-to-end QA set

**Chosen.** Action Genome + VidOR for extraction; ~100 hand-authored QA pairs for the rest.

| | |
|---|---|
| **Division of labor** | Public sets encode per-frame predicate **states**, not events — so they **cannot** validate the reducer. Their clips are too short to exercise long-horizon Re-ID. Both gaps are precisely what the QA set covers. The two sources are complementary, not redundant. |
| **Why both datasets** | Action Genome is the de facto VidSGG benchmark, so extraction numbers are comparable to published work; its indoor-activity ontology maps poorly to other domains. VidOR supplies object *trajectories*, which double as tracklet ground truth for Re-ID, and longer horizons. Cost is two loaders and two ontology mappings. |
| **Also rejected** | Component metrics only (MOT17/DanceTrack + AG) — cheap and standard, but nothing measures whether the assembled pipeline answers a question correctly, which is the only promise the SDK makes. |
| **Also rejected** | Synthetic video (Kubric/Blender) with perfect labels — free, unlimited, and directly tunes thresholds, but sim-to-real gap means thresholds tuned on clean renders will not transfer to compressed CCTV. Useful later for reducer unit fixtures. |
| **Hard dependency** | D3's probe threshold and the Phase-2 similarity threshold are **untunable without this**. It is a v0 prerequisite, not a follow-up. |

---

### D10 — States are reified interval nodes, not relationship properties

**Chosen.** `(:Entity)-[:SUBJECT_OF]->(:State {type, valid_from, valid_until, ...})-[:OBJECT_OF]->(:Entity)`

| | |
|---|---|
| **Runner-up** | `valid_from` / `valid_until` as properties on the relationship (the v2.0.0 position) |
| **Why not** | Three concrete problems. (a) Neither Neo4j nor Memgraph indexes relationship properties as well as node properties, so every temporal filter degrades to a scan over matched edges. (b) Two concurrent instances of the same relation between the same pair collide — you cannot represent "held the bag, put it down, picked it up again" as two edges of the same type without losing one. (c) Provenance (source frame, model, confidence) has nowhere clean to live. |
| **Also rejected** | Event nodes only, intervals computed per query — ingest becomes trivially idempotent and re-runnable under new reducer rules with no migration. Genuinely attractive. Rejected because every query then pays reduction cost and correctness lives in the compiler rather than in the data, making the store un-inspectable. |
| **Also rejected** | Both: append-only event log as source of truth, State nodes as a droppable materialized view — the cleanest and most defensible design, and the likely v1 target. Rejected for v0 only as scope. |
| **Cost accepted** | One extra hop per query and uglier Cypher — but the IR compiler writes it, so the ugliness is never user-facing. |

---

### D11 — Content-addressed on-disk stage cache

**Chosen.** Each stage writes an artifact keyed by `hash(video_content_hash, stage_name, stage_config)`.

| | |
|---|---|
| **Motivation** | With a hosted VLM, extraction costs real money. A crash at stage 4 must not re-bill stage 3. More importantly, the eval loop requires changing one stage's config and re-running only downstream of it — otherwise tuning is unaffordable and therefore will not happen. |
| **Runner-up** | Persist only the VLM responses, keyed by frame hash + prompt hash |
| **Why not** | Captures ~90% of the cost benefit for ~20% of the work — but changing the **filter** changes which frames exist, so the keys miss and the entire VLM cost is re-billed. Since the filter is the component most in need of tuning, this optimizes the wrong axis. |
| **Also rejected** | Monolithic in-memory run — simplest code, but any tweak re-bills everything. |
| **Also rejected** | Use the graph as the cache and resume by querying it — no separate format, but it places half-finished pre-Re-ID data in the queryable store, so users can query a graph that is silently wrong. Unacceptable given the whole point is trustworthy answers. |
| **Cost accepted** | A small cache layer, and disk footprint dominated by object crops. |

---

### D12 — Set-of-Mark: numbered boxes drawn on the image

**Chosen.** Render each tracked box on the frame with its `local_id` as a visible label; ask for relations between numbered ids.

| | |
|---|---|
| **Runner-up** | Clean image plus a text list `7: person [x,y,w,h]` |
| **Why not** | No image mutation and trivial to implement, but the model must ground coordinates to regions itself — which is exactly the weakness that motivated taking detection away from it in D4. Reintroducing it at the reference-resolution step defeats D4. |
| **Also rejected** | Both marks and coordinates — redundant grounding usually helps and is cheap given the renderer exists. Worth A/B testing; costs tokens, and creates a conflict case when a mark is occluded but coordinates disagree. Flagged as an experiment, not a default. |
| **Also rejected** | Per-entity crops with ids — maximally unambiguous per entity, but destroys the spatial context that inter-entity relations depend on. Structurally wrong for relation extraction. |
| **Cost accepted** | An overlay renderer, label-occlusion handling in crowded frames, and the marks slightly perturb the pixels the model sees. |

---

### D13 — Embedded Kuzu is the default store; Memgraph is an opt-in adapter

**Chosen.** Quickstart runs with no Docker, no service, no ports.

| | |
|---|---|
| **Motivation** | Docker was the second-largest quickstart wall after the GPU requirement removed in D2. Removing both means the README is executable by its reader. |
| **Enabled by D8** | We own all generated Cypher, so a dialect is a *compiler target*, not a user-facing concern. Backend portability is unusually cheap here. |
| **Runner-up** | Memgraph via bundled docker-compose (the v2.0.0 position) |
| **Why not** | One backend, one dialect, in-memory speed, full Cypher including variable-length paths, and it matches the VPC deployment model. But every user hits a Docker dependency before line one, and CI needs a service container. Kept as the production adapter. |
| **Also rejected** | Neo4j — largest ecosystem and an Aura free tier that also avoids local Docker; slower on multi-hop and community-edition limits. Third backend, later. |
| **Also rejected** | NetworkX in-memory for v0 — zero deps and trivially testable, but it deletes Cypher entirely, leaving the IR compiler (the highest-design-risk component) unvalidated and destined for a rewrite. |
| **Cost accepted** | Two Cypher dialects from day one. Kuzu's mandatory typed schema is actually a *fit*, since D7 already gives us a declared schema to generate DDL from. |

---

### D14 — OpenRouter as the single VLM gateway, behind a `VLMBackend` protocol

**Chosen.** One key, one API, many vision models.

| | |
|---|---|
| **Motivation** | The eval matrix is the point. Comparing extraction quality across five vision models should be a config change, not five SDK integrations. |
| **Runner-up** | Provider-agnostic protocol with a first-party SDK default (e.g. Claude via tool-use schema) |
| **Why not** | Strongest per-provider conformance guarantees and one fewer hop. Rejected because it makes cross-model comparison — the thing that de-risks D4 and D12 — expensive. |
| **Also rejected** | Gemini default with native video input — would give a genuinely useful whole-video baseline ablation, but native long-context video *undercuts the premise* that keyframe pruning is necessary. That is a claim to defend with numbers, not to design around. Keep as an ablation backend. |
| **Costs accepted** | (a) Structured-output support varies per underlying model, so conformance is best-effort plus validation, not guaranteed. (b) An extra failure surface on a core dependency. (c) **It makes the default path network-dependent, contradicting the air-gapped claim in §10.4** — SGLang is the air-gap answer, not a performance option. |
| **Key contrast** | SGLang + XGrammar guarantees grammar conformance *by construction*. Hosted APIs give schema-constrained tool use, which is strong but not a hard guarantee. The protocol must abstract over two genuinely different enforcement mechanisms and must handle validation failure rather than assume it away. |

---

### D15 — Non-conformant extraction: bounded retry, then drop the frame and record it

**Chosen.** Retry with the Pydantic validation error fed back; on exhaustion, skip the frame and append the raw response to `run_manifest.json`. `graph.report()` surfaces the drop rate.

| | |
|---|---|
| **Runner-up** | Fail the run when drop rate exceeds a threshold |
| **Why not** | Strongest correctness stance — it prevents building a graph nobody should trust. Rejected as a *default* because an 8-hour ingest dying at 70% is infuriating. **Adopted as an opt-in guard** (`max_drop_rate`), which the stage cache makes tolerable. |
| **Also rejected** | Partial acceptance: keep valid entities, drop invalid relations — maximizes recall from a flaky model, but a delta with entities and no relations contributes nothing to the graph, and partial data makes the drop-rate metric meaningless. |
| **Also rejected** | Strict-mode allowlist of models with verified structured-output support — prevents the failure class rather than handling it, but the table rots as models change and it blocks legitimate experimentation. **Adopted as a warning, not a gate.** |
| **Cost accepted** | The graph has holes. A systematically failing entity type yields a plausible-looking but incomplete graph — which is why the drop rate must be loud, per-class, and in the default output. |

---

## 3. Architecture

### 3.1 End-to-end flow

Six stages, strictly ordered, each cache-keyed. Batch only (D5).

```
                              video.mp4
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ STAGE 1 — DualPhaseFilter                            (CPU, cheap) │
│                                                                   │
│   demux ──► MV side-data gate ──► global-motion probe ──►         │
│   [ego-motion comp] ──► decode survivors ──► embedding dedupe     │
│                                                                   │
│   in:  ~216,000 frames (2h @ 30fps)                               │
│   out: ~1,500 keyframes                        keyframes.parquet  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ STAGE 2 — Detect + Track                          (CPU/GPU, mid)  │
│                                                                   │
│   detector (YOLOv11 / RT-DETR) ──► ByteTrack ──► tracklets        │
│                                                                   │
│   out: boxes + local_id + crops                tracklets.parquet  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ STAGE 3 — VLMExtractor                    (network, EXPENSIVE $$) │
│                                                                   │
│   Set-of-Mark overlay ──► OpenRouter ──► Pydantic validate ──►    │
│   [retry ≤N with error fed back] ──► SemanticDelta | DROP         │
│                                                                   │
│   out: events                    deltas.jsonl + run_manifest.json │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ STAGE 4 — GlobalReID  (offline)                   (GPU-ish, mid)  │
│                                                                   │
│   OSNet per crop ──► per-tracklet robust embedding ──►            │
│   constrained agglomerative clustering ──► canonical ids          │
│                                                                   │
│   out: local_id → canonical_id             canonical_map.parquet  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ STAGE 5 — StateReducer                            (CPU, trivial)  │
│                                                                   │
│   rewrite ids ──► sort by timestamp ──► replay transition rules   │
│   ──► materialize intervals                                       │
│                                                                   │
│   out: (subj, rel, obj, valid_from, valid_until)  states.parquet  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│ STAGE 6 — GraphStore.write                        (CPU, trivial)  │
│                                                                   │
│   generate DDL from schema ──► bulk upsert Entity/State/Event     │
│                                                                   │
│   out: Kuzu database (or Memgraph)                                │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
                          ┌───────────────┐
      "Did he drop the    │   RETRIEVAL   │
       bag before he  ───►│               │
       left?"             │  NL → IR tree │
                          │  → validate   │
                          │  → compile    │
                          │  → Cypher     │
                          └───────┬───────┘
                                  ▼
                              answer + provenance
```

### 3.2 Where the risk lives

```
 risk
  ▲
  │  ┌────────────────┐
  │  │ StateReducer   │◄── novel, no public GT, silent failures
  │  │ + IR compiler  │
  │  └────────────────┘
  │        ┌──────────────┐
  │        │ GlobalReID   │◄── irreversible errors, needs VidOR GT
  │        └──────────────┘
  │              ┌────────────────┐
  │              │ VLM extraction │◄── measurable against AG
  │              └────────────────┘
  │                    ┌───────────────┐
  │                    │ Filter        │◄── silent recall loss
  │                    └───────────────┘
  │                          ┌──────────────┐
  │                          │ Detect/Track │◄── solved problem
  │                          └──────────────┘
  └──────────────────────────────────────────────► maturity
```
v0 deliberately front-loads the top-left quadrant (reducer + IR) because it is novel and has no off-the-shelf validation.

---

## 4. Module Specifications

### 4.1 `visiongraph.preprocessing` — DualPhaseFilter

**Purpose:** eliminate inter-frame redundancy before the expensive stage, so VLM spend scales with *semantic* change rather than with wall-clock duration.

#### Phase 1 — compressed-domain motion gate

Read macroblock motion vectors from P-frames via PyAV side data (`AV_FRAME_DATA_MOTION_VECTORS`, enabled by the `export_mvs` codec flag). No pixel decode required for the gate itself.

Motion vectors are **not** optical flow. Corrections required before any fit:

1. **Reference-distance normalization.** A vector referencing a frame 3 steps back represents 3× the per-frame displacement of one referencing the immediately previous frame. Normalize: `v_norm = v_raw / (dst_poc - src_poc)`.
2. **Missing-data handling.** Skipped and intra-coded macroblocks carry no vector. Treat them as *missing*, never as zero — zeroing them biases every global fit toward "static".
3. **Rate-distortion noise.** Encoders select vectors to minimize bitrate, not to describe physical motion. In low-texture regions the chosen vector is near-arbitrary. Weight blocks by residual energy where available.

```
frame k (P):  MV field, 16×16 blocks
  ┌───┬───┬───┬───┬───┬───┐
  │ → │ → │ → │ ↗ │ ✗ │ → │   → global pan (all blocks agree)
  ├───┼───┼───┼───┼───┼───┤   ↗ local motion (outlier vs global)
  │ → │ → │ ↗ │ ↗ │ → │ → │   ✗ skipped/intra — MISSING, not zero
  ├───┼───┼───┼───┼───┼───┤
  │ → │ → │ → │ → │ → │ → │
  └───┴───┴───┴───┴───┴───┘
         │
         ├──► RANSAC affine fit  ──►  global transform G_k
         │
         └──► residual r = v_norm − G_k·p   ──► localized motion energy
```

#### Global-motion probe (auto-detect, per D3)

On a probe window (default: 300 frames sampled across the file), fit `G_k` and compute `median(||G_k||)`. Below a threshold, the camera is static: **bypass compensation entirely** and threshold raw MV magnitude. Above it, compensation runs. This is what makes "both modalities first-class" affordable — a static camera pays nothing for machinery it does not need.

#### Phase 2 — embedding delta

Survivors are decoded and embedded with a small CLIP or DINO model. Drop if `cos(e_i, e_last_kept) > τ_embed`. This absorbs residual panning artifacts that Phase 1 misses and provides a flatter, more transferable threshold response than raw motion energy.

#### Interface

```python
@dataclass(frozen=True)
class FilterConfig:
    motion_threshold: float = 0.4        # residual MV energy, needs eval-set tuning
    embed_threshold: float = 0.92        # cosine similarity above which a frame is redundant
    use_ego_motion: Literal["auto", "always", "never"] = "auto"
    probe_frames: int = 300
    max_gap_seconds: float = 5.0         # force-keep a frame at least this often

class DualPhaseFilter:
    def select(self, video_path: Path) -> KeyframeSet: ...
```

`max_gap_seconds` is a safety net: it guarantees a keyframe floor so a mis-tuned threshold degrades sampling density rather than producing a hours-long hole.

#### Failure modes

| Failure | Symptom | Mitigation |
|---|---|---|
| HEVC yields no MVs (**⚠ VERIFY**) | Phase 1 keeps everything or nothing | Detect empty MV side data; fall back to decode-and-embed with a logged warning |
| Threshold too aggressive | Missing graph edges, no error | `max_gap_seconds` floor + kept-frame recall metric vs AG annotations |
| PTZ camera intermittently pans | Bursts of false positives on each pan | `auto` probe re-evaluated per chunk, not once per file |

---

### 4.2 `visiongraph.detect` — Detector + Tracker

**Purpose:** produce metrically reliable boxes and deterministic within-tracklet identity (D4).

- **Detector:** YOLOv11n / RT-DETR-S by default. At post-prune frame rates (~1,500 frames for a 2h video) this is CPU-viable, which is what preserves the zero-GPU quickstart from D2.
- **Tracker:** ByteTrack. Chosen because it associates low-confidence detections against existing tracks, which is exactly the occlusion case that fragments tracklets and therefore multiplies Re-ID work downstream.
- **Output:** one row per (frame, local_id) with box, class, confidence, plus a persisted crop for Stage 4.

A **tracklet** is a maximal run of frames sharing one `local_id`. Tracklet boundaries are where identity uncertainty is introduced, and thus exactly the seams Stage 4 must sew.

```
time ──────────────────────────────────────────────────────────►
local_id 1  ████████████
local_id 2       ██████████████
local_id 3                        ███████         ← occlusion split
local_id 4                                ██████    same person as 3?
local_id 5                    ████████████████████
                                  ▲
                          Stage 4's entire job
```

#### Interface

```python
@dataclass(frozen=True)
class DetectConfig:
    detector: str = "yolo11n"
    label_set: tuple[str, ...] = ("person", "bag", "vehicle", ...)   # required by D4
    conf_threshold: float = 0.25
    track_buffer: int = 30               # frames a lost track survives before termination
    save_crops: bool = True              # required for Stage 4
```

---

### 4.3 `visiongraph.extractors` — VLMExtractor

**Purpose:** the only generative component. Emits relations between *given* entity ids. It never invents entities and never emits coordinates (D4).

#### Backend protocol

Two implementations from day one, per D1:

```python
class VLMBackend(Protocol):
    def extract(self, image: Image, schema: type[BaseModel], prompt: str) -> RawResponse: ...

class OpenRouterBackend(VLMBackend): ...   # default (D14)
class SGLangBackend(VLMBackend): ...       # local / air-gapped, XGrammar-enforced
```

The protocol must abstract two genuinely different conformance mechanisms:

| Backend | Mechanism | Guarantee |
|---|---|---|
| SGLang + XGrammar | Constrained decoding against the grammar | **Hard** — malformed output is unrepresentable |
| OpenRouter | Schema-as-tool-input | **Best-effort** — varies by underlying model; validation + retry required |

This asymmetry is why D15's failure policy is a first-class part of the design rather than an afterthought.

#### Set-of-Mark prompting (D12)

```
    raw frame                       annotated frame sent to VLM
 ┌───────────────────┐            ┌───────────────────┐
 │      🧍  🧍       │            │  ┌─1─┐   ┌─2─┐    │
 │           👜      │    ───►    │  │🧍 │   │🧍 │     │
 │   🚪              │            │  └───┘ ┌3┐└───┘    │
 │                   │            │ ┌4┐    │👜│        │
 └───────────────────┘            │ │🚪│   └─┘         │
                                  │ └─┘               │
                                  └───────────────────┘

 prompt: "Entities 1,2 are person; 3 is bag; 4 is door.
          Report only relationships between these numbered ids."
```

Renderer requirements: labels placed outside the box where space allows, deterministic color assignment per id (so the same entity looks the same across frames), and a crowding policy — above N boxes, split into multiple calls by spatial region rather than producing an unreadable frame.

#### Extraction contract

```python
class Entity(BaseModel):
    local_id: str
    class_label: str
    bbox: list[int] = Field(description="[x_min, y_min, x_max, y_max]")
    # NOTE: populated by the detector, echoed for provenance — NOT generated by the VLM

class TemporalEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str = Field(description="Action verb from the schema vocabulary, e.g. DROPS, ENTERS")
    timestamp: float

class SemanticDelta(BaseModel):
    frame_type: str = "P-Node"
    timestamp: float
    entities: list[Entity]
    relationships: list[TemporalEdge]
```

`relation` is validated against the schema's declared vocabulary. An out-of-vocabulary verb is a validation error and triggers retry — this is the mechanism that keeps the ontology closed.

#### Retry and drop policy (D15)

```
attempt 1 ──► pydantic.ValidationError ──┐
                                          │  error text appended to prompt
attempt 2 ◄───────────────────────────────┘
   │
   ├── valid  ──► deltas.jsonl
   └── invalid after N ──► DROP
                            │
                            └──► run_manifest.json
                                  { timestamp, raw_response, error,
                                    entity_classes_present }
```

`graph.report()` surfaces overall drop rate **and drop rate per entity class**. The per-class breakdown is the point: a uniform 3% drop is noise, while 60% of frames containing `vehicle` failing is a schema bug that a single aggregate number would hide.

Opt-in guard: `max_drop_rate` aborts the run when exceeded (the rejected-then-adopted runner-up from D15).

---

### 4.4 `visiongraph.cluster` — GlobalReID

**Purpose:** map fragmented tracklets to canonical entities so duration and history accumulate against one node (D5).

#### Algorithm

1. **Per-crop embedding.** OSNet over each persisted crop.
2. **Crop quality scoring.** Score by box area, detector confidence, and blur (variance of Laplacian). Small, low-confidence, or motion-blurred crops produce unreliable appearance vectors.
3. **Robust tracklet embedding.** Trimmed mean over the top-k quality crops — not a plain mean, which a few occluded frames can drag arbitrarily.
4. **Constrained agglomerative clustering** over tracklet embeddings with a **cannot-link** constraint: two tracklets whose frame ranges overlap cannot be the same entity, because one entity cannot be in two places at one instant.

```
tracklet timeline                     constraint graph
 A ████████                            A ──cannot── B   (overlap)
 B    ████████                         A ──cannot── C   (overlap)
 C    ██████                           B ──cannot── C   (overlap)
 D              ████████               D  free
 E                  ██████             D ──cannot── E   (overlap)

 embedding distance:  d(A,D)=0.08  d(B,E)=0.11  d(A,B)=0.09
   ⚠ d(A,B) is small, but A│B overlap ⇒ cannot-link ⇒ NOT merged
   ✓ A+D merged → canonical_1
   ✓ B+E merged → canonical_2
```

The constraint is what makes this materially better than greedy matching: appearance similarity alone would have fused A and B, and greedy online matching had no mechanism to refuse.

5. **Emit** `canonical_map.parquet`: `local_id → canonical_id`, plus the merge distances for review.

#### Interface

```python
@dataclass(frozen=True)
class ReIDConfig:
    model: str = "osnet_x1_0"
    similarity_threshold: float = 0.85     # agglomerative cut height
    top_k_crops: int = 16
    trim_fraction: float = 0.2
    vector_store: Literal["lancedb", "qdrant", "memory"] = "lancedb"
```

#### Failure modes

| Failure | Consequence | Mitigation |
|---|---|---|
| Cut height too low | Node explosion — one person becomes 30 entities | Report cluster-count vs tracklet-count ratio |
| Cut height too high | Two people fused irreversibly | Cannot-link constraint blocks the temporally-impossible subset; merge distances exported for review |
| Non-person classes | OSNet is trained for person Re-ID; bag/vehicle embeddings are weak | v0: only apply Re-ID to configured classes; others stay tracklet-scoped |

That last row is a real limitation to note at review: **appearance Re-ID for generic objects is unsolved**, and pretending otherwise would silently corrupt object identity.

---

### 4.5 `visiongraph.schema` — Schema-as-Code

**Purpose:** make the ontology an explicit, diffable, testable artifact (D6, D7).

```python
from visiongraph.schema import SkillSchema, DurativeRelation, InstantRelation

WAREHOUSE = SkillSchema(
    name="warehouse",
    entity_classes=("person", "bag", "forklift", "door", "pallet"),

    instant_relations=(
        InstantRelation("PICKS_UP",  subject="person", object=("bag", "pallet")),
        InstantRelation("DROPS",     subject="person", object=("bag", "pallet")),
        InstantRelation("HANDS_OFF", subject="person", object=("bag", "pallet")),
        InstantRelation("ENTERS",    subject="person", object="door"),
        InstantRelation("EXITS",     subject="person", object="door"),
    ),

    durative_relations=(
        DurativeRelation(
            "POSSESSES",
            subject="person", object=("bag", "pallet"),
            opens_on=("PICKS_UP",),
            closes_on=("DROPS", "HANDS_OFF"),
            cardinality="one_subject_per_object",
        ),
        DurativeRelation(
            "INSIDE",
            subject="person", object="door",
            opens_on=("ENTERS",), closes_on=("EXITS",),
            cardinality="unconstrained",
        ),
    ),
)
```

Three things this buys:

1. The relation vocabulary constrains VLM output (§4.3), keeping the ontology closed.
2. `opens_on` / `closes_on` drive the reducer (§4.6) deterministically.
3. `entity_classes` and relation typing generate Kuzu DDL (§5.3) and populate the IR's valid type vocabulary (§6), so an invalid type name is a *validation error* rather than an empty result.

#### Scaffolding CLI (D7)

```
$ visiongraph schema scaffold --entities person,bag,door --relations PICKS_UP,DROPS,ENTERS,EXITS
  → LLM proposes durative relations and their transition tables
  → writes schemas/warehouse.py for human review
  → writes tests/test_warehouse_schema.py with reducer fixtures
```

Inference happens once, at authoring time. Runtime is deterministic. The generated test file matters as much as the schema: it turns proposed semantics into assertions.

---

### 4.6 `visiongraph.reduce` — StateReducer

**Purpose:** convert the instantaneous event stream into interval-valid states. Closes the v2.0.0 hole identified in D6.

Pure function of `(events, rules) → states`. Zero VLM involvement, therefore fully unit-testable on hand-written fixtures — which is why v0 can validate it before extraction quality is known.

#### Semantics

```
events (after canonical id rewrite, sorted by timestamp)

t=3.1   P1 PICKS_UP  B1
t=8.0   P1 ENTERS    D1
t=12.4  P1 DROPS     B1
t=19.2  P1 EXITS     D1
t=25.0  P2 PICKS_UP  B1

                        ▼ replay against rules

POSSESSES(P1,B1)  ├────────────────┤            valid_from=3.1  valid_until=12.4
INSIDE(P1,D1)             ├──────────────┤      valid_from=8.0  valid_until=19.2
POSSESSES(P2,B1)                              ├──────►  valid_from=25.0  valid_until=NULL

 timeline ──┬────┬────┬────┬────┬────┬────┬────┬────►
            3.1  8.0  12.4 19.2 25.0            EOF
```

`valid_until = NULL` means "still open at end of video" — explicitly *not* "unknown". Query semantics must treat open intervals as extending to the observation horizon, which is why every generated Cypher predicate includes the `IS NULL` branch (the exact off-by-one class that D8 rejected free-form generation over).

#### Edge-case policy (resolves open item #3 from v2.0.0)

| Case | Default policy | Rationale |
|---|---|---|
| `closes_on` event with no open interval | Emit a zero-length state and flag it | Loses no information; the flag surfaces upstream extraction error rather than hiding it |
| Second `opens_on` while already open | Ignore; extend the existing interval | A repeated observation of an ongoing state is the common case |
| `cardinality="one_subject_per_object"` violated | Auto-close the prior holder's interval at the new open | Physically correct: a handoff implies the previous holder released |
| `cardinality="unconstrained"` overlap | Keep both, no resolution | Legitimate multi-valued state |
| Interval open at EOF | `valid_until = NULL` | Distinguishes "still true" from "ended" |
| Contradictory events at identical timestamp | Deterministic tie-break by (relation name, subject id) and flag | Determinism is non-negotiable; the flag makes it auditable |

All flags land in the run manifest and in `graph.report()`. Every one of these is a **policy**, not an inference — which is what makes the reducer reviewable.

#### Interface

```python
class StateReducer:
    def __init__(self, schema: SkillSchema): ...
    def reduce(self, events: Iterable[TemporalEdge], horizon: float) -> list[State]: ...
```

---

## 5. Graph Data Model

### 5.1 Reified interval nodes (D10)

```
        ┌──────────────────┐                      ┌──────────────────┐
        │ Entity           │                      │ Entity           │
        │ id: canonical_1  │                      │ id: canonical_7  │
        │ class: person    │                      │ class: bag       │
        └────────┬─────────┘                      └─────────┬────────┘
                 │                                          │
        SUBJECT_OF│                                         │OBJECT_OF
                 │        ┌──────────────────────┐          │
                 └───────►│ State                │◄─────────┘
                          │ type: POSSESSES      │
                          │ valid_from:  3.1     │
                          │ valid_until: 12.4    │
                          │ confidence:  0.87    │
                          │ src_frames: [93,372] │
                          │ model: <backend/id>  │
                          └──────────────────────┘

        ┌──────────────────┐
        │ Event            │      also stored: the raw instantaneous
        │ type: DROPS      │      observations, so the graph remains
        │ timestamp: 12.4  │      inspectable and the reducer's output
        │ src_frame: 372   │      is checkable against its input
        └──────────────────┘
```

### 5.2 Why not properties on the edge

```
  REJECTED (v2.0.0)                       CHOSEN (D10)

  (P1)-[:POSSESSES {                      (P1)-[:SUBJECT_OF]->(:State{
        valid_from: 3.1,                        type:'POSSESSES',
        valid_until: 12.4                       valid_from:3.1,
      }]->(B1)                                  valid_until:12.4
                                            })-[:OBJECT_OF]->(B1)

  ✗ relationship properties poorly        ✓ node properties are indexed
    indexed → temporal filter scans         → range scan on valid_from
  ✗ P1 picks up B1 twice ⇒ two            ✓ two State nodes, both retained
    identical edges, one is lost
  ✗ provenance has no home                ✓ provenance is just more props
  ✓ one hop shorter, prettier Cypher      ✗ one extra hop (compiler-hidden)
```

### 5.3 Kuzu DDL, generated from the schema

Kuzu requires an explicit typed schema. Since D7 already produces one, this is generation, not extra burden:

```sql
CREATE NODE TABLE Entity (id STRING, class STRING, first_seen DOUBLE,
                          last_seen DOUBLE, PRIMARY KEY(id));
CREATE NODE TABLE State  (sid STRING, type STRING, valid_from DOUBLE,
                          valid_until DOUBLE, confidence DOUBLE,
                          src_frames STRING, flags STRING, PRIMARY KEY(sid));
CREATE NODE TABLE Event  (eid STRING, type STRING, timestamp DOUBLE,
                          src_frame INT64, PRIMARY KEY(eid));

CREATE REL TABLE SUBJECT_OF (FROM Entity TO State);
CREATE REL TABLE OBJECT_OF  (FROM State  TO Entity);
CREATE REL TABLE EVENT_SUBJ (FROM Entity TO Event);
CREATE REL TABLE EVENT_OBJ  (FROM Event  TO Entity);
CREATE REL TABLE DERIVED_FROM (FROM State TO Event);
```

`DERIVED_FROM` is the audit edge: it links every materialized interval back to the events that opened and closed it, so a wrong interval is traceable to a wrong extraction rather than being an unexplained fact.

### 5.4 Store protocol

```python
class GraphStore(Protocol):
    def init_schema(self, schema: SkillSchema) -> None: ...
    def write(self, entities, states, events) -> None: ...
    def execute(self, cypher: str, params: dict) -> list[dict]: ...
    @property
    def dialect(self) -> Dialect: ...        # consumed by the compiler

class KuzuStore(GraphStore): ...       # default, embedded (D13)
class MemgraphStore(GraphStore): ...   # opt-in, production
```

Writes are idempotent upserts keyed on `sid`/`eid` derived from content hashes, so re-running Stage 6 is safe.

---

## 6. Retrieval: Temporal IR and Cypher Compiler

### 6.1 Why an IR (D8)

Question *types* are domain-independent; only vocabulary is domain-specific. So the operator set is small and closed, expressiveness comes from composition, and the schema supplies type names.

```
 "Did P1 drop the bag before leaving?"
                 │
                 ▼  LLM (the only non-deterministic step)
        ┌────────────────────┐
        │ IR tree (Pydantic) │
        └────────┬───────────┘
                 ▼  validate against SkillSchema
        invalid type name ──► LOUD ERROR (never an empty result)
                 │
                 ▼  compiler (deterministic, unit-tested per dialect)
             Cypher + params
                 │
                 ▼
             store ──► rows ──► answer + provenance
```

### 6.2 Operator set

```python
class HoldsAt(BaseModel):     # was state S true at instant t?
    op: Literal["holds_at"]; relation: str; subject: str | None
    object: str | None; t: float

class HoldsDuring(BaseModel): # was S true throughout / at any point in [t0,t1]?
    op: Literal["holds_during"]; relation: str; subject: str | None
    object: str | None; t0: float; t1: float; mode: Literal["all", "any"] = "any"

class EventAt(BaseModel):     # did event E occur in [t0,t1]?
    op: Literal["event_at"]; relation: str; subject: str | None
    object: str | None; t0: float | None = None; t1: float | None = None

class Precedes(BaseModel):    # did A happen before B?
    op: Literal["precedes"]; a: "IRNode"; b: "IRNode"

class Overlaps(BaseModel):    # do two intervals intersect?
    op: Literal["overlaps"]; a: "IRNode"; b: "IRNode"

class Count(BaseModel):
    op: Literal["count"]; inner: "IRNode"

class PathBetween(BaseModel): # multi-hop relational reachability
    op: Literal["path_between"]; source: str; target: str
    max_hops: int = 3; at: float | None = None

class And(BaseModel): op: Literal["and"]; children: list["IRNode"]
class Or(BaseModel):  op: Literal["or"];  children: list["IRNode"]
class Not(BaseModel): op: Literal["not"]; child: "IRNode"

IRNode = Annotated[Union[HoldsAt, HoldsDuring, EventAt, Precedes,
                         Overlaps, Count, PathBetween, And, Or, Not],
                   Field(discriminator="op")]
```

Eight primitives plus three combinators. `relation` and entity `class` values are validated against the active `SkillSchema`, which is the mechanism that converts a would-be silent empty result into a raised error.

### 6.3 Lowering examples

`HoldsAt` — the interval predicate, written once and tested, rather than regenerated per query:

```cypher
MATCH (s:Entity)-[:SUBJECT_OF]->(st:State)-[:OBJECT_OF]->(o:Entity)
WHERE st.type = $relation
  AND ($subject IS NULL OR s.id = $subject)
  AND ($object  IS NULL OR o.id = $object)
  AND st.valid_from <= $t
  AND (st.valid_until IS NULL OR st.valid_until > $t)
RETURN s.id AS subject, o.id AS object,
       st.valid_from, st.valid_until, st.confidence, st.src_frames
```

The `valid_until IS NULL` branch and the `<=` / `>` asymmetry are the two details free-form generation gets wrong most often — half-open intervals `[from, until)` mean an instant equal to `valid_until` is *outside* the state.

`Precedes(EventAt(DROPS), EventAt(EXITS))` — the question from §6.1:

```cypher
MATCH (s1:Entity)-[:EVENT_SUBJ]->(e1:Event)
WHERE e1.type = $rel_a AND s1.id = $subj_a
MATCH (s2:Entity)-[:EVENT_SUBJ]->(e2:Event)
WHERE e2.type = $rel_b AND s2.id = $subj_b
  AND e2.timestamp > e1.timestamp
RETURN e1.timestamp AS a_at, e2.timestamp AS b_at,
       e1.src_frame AS a_frame, e2.src_frame AS b_frame
ORDER BY a_at LIMIT 1
```

### 6.4 Result contract

Never a bare boolean. A wrong-looking answer must be inspectable:

```python
@dataclass(frozen=True)
class QueryResult:
    answer: bool | int | list[dict]
    ir: IRNode                  # what was actually asked
    cypher: str                 # what actually ran
    rows: list[dict]            # raw evidence
    provenance: list[int]       # source frame indices
    warnings: list[str]         # e.g. "3 states in range carry reducer flags"
```

Returning `ir` and `cypher` is deliberate: the NL step is the only non-deterministic component in the retrieval path, so its output must be visible for the answer to be trustworthy.

### 6.5 Expressiveness ceiling

Out-of-algebra questions must fail loudly, not degrade:

- Aggregations over unbounded groupings ("busiest hour") — needs a `group_by` operator, deferred.
- Counterfactuals ("what if the door had been locked") — out of scope permanently.
- Appearance queries ("the man in the red jacket") — needs attribute extraction in the schema; a v1 extension.

---

## 7. Cache and Resumability (D11)

### 7.1 Key derivation

```
key(stage) = sha256(
    video_content_hash          ‖   # not path, not mtime
    stage_name                  ‖
    canonical_json(stage_config)‖   # incl. model ids and prompt template hash
    key(upstream_stage)             # chained ⇒ invalidation cascades
)
```

Chaining upstream keys is what makes invalidation correct: changing the filter *must* invalidate extraction, because it changes which frames exist.

### 7.2 Invalidation cascade

```
change                        invalidates                          $ cost
─────────────────────────────────────────────────────────────────────────
reducer rule                  stage 5, 6                           free
cut height (Re-ID)            stage 4, 5, 6                        free
VLM prompt / model            stage 3, 4, 5, 6                     FULL $$
detector / label_set          stage 2, 3, 4, 5, 6                  FULL $$
filter threshold              stage 1..6                           FULL $$
```

Read the table as a tuning-order guide: iterate cheapest-first. Reducer and IR work — the highest-risk components per §3.2 — are also the free ones, which is a fortunate alignment and another reason v0 targets them first.

### 7.3 Artifacts

| Stage | Artifact | Schema |
|---|---|---|
| 1 | `keyframes.parquet` | `frame_idx, pts, timestamp, keep_reason, motion_energy, embed_dist` |
| 2 | `tracklets.parquet` | `frame_idx, local_id, class, bbox, conf, crop_path, quality` |
| 3 | `deltas.jsonl` | one `SemanticDelta` per line |
| 3 | `run_manifest.json` | drops, retries, per-class drop rate, token/cost totals |
| 4 | `canonical_map.parquet` | `local_id, canonical_id, merge_distance, cluster_size` |
| 5 | `states.parquet` | `sid, type, subject, object, valid_from, valid_until, flags` |

Parquet for columnar/typed stages, JSONL for the nested one. `keep_reason` in Stage 1 exists specifically so filter behavior is debuggable after the fact.

---

## 8. v0 Scope and Milestones

**Definition of done:** one short video in → entity and state nodes in Kuzu → an answerable temporal query out.

### In scope

Fixed-stride sampling · YOLOv11n + ByteTrack · Set-of-Mark overlay · OpenRouter extraction with retry/drop · one hardcoded schema · StateReducer with full edge-case policy · reified Kuzu store · IR compiler (Kuzu dialect) · content-addressed stage cache · ~20 QA pairs on Action Genome clips.

### Deferred, with reasons

| Deferred | Why it is safe to defer |
|---|---|
| MV / embedding filter | Fixed stride is correct-but-wasteful; on short clips the cost is trivial |
| Ego-motion compensation | Blocked on ⚠ VERIFY and on the eval set |
| GlobalReID | AG clips are short enough that tracklet ≈ entity; deferring is principled, not lazy |
| SGLang backend | Second backend proves the protocol; not needed to prove the architecture |
| `schema scaffold` CLI | One hand-written schema suffices for one domain |
| Memgraph dialect | Kuzu validates that the compiler works at all |
| MCP / FastAPI | Pure surface area over a proven core |
| VidOR loader | Only needed once Re-ID exists |

### Milestone order

```
M1  Schema + StateReducer + fixtures        no video, no VLM, no money
      └─ prove the novel semantics in unit tests first

M2  Kuzu store + DDL generation + IR compiler (Kuzu)
      └─ hand-written states in, correct query answers out

M3  Detect + Track + Set-of-Mark renderer
      └─ visually inspect annotated frames before spending a cent

M4  OpenRouter extraction + retry/drop + manifest
      └─ first real deltas; eyeball them before wiring the graph

M5  End-to-end fit() + stage cache
      └─ DEFINITION OF DONE

M6  AG loader + 20 QA pairs + metric harness
      └─ the first number anyone should believe
```

M1–M2 cost nothing and de-risk the components with no external ground truth. M3 gates spend on a visual check. M4 is the first dollar spent — deliberately fourth.

---

## 9. Evaluation

### 9.1 Ground-truth allocation (D9)

```
                    ┌───────────────────────────────────────────┐
 Action Genome ────►│ per-frame entities + predicates            │
                    │ frame-filter recall                        │
                    │ ✓ published-comparable                     │
                    │ ✗ cannot validate the reducer (no events)  │
                    │ ✗ clips too short for Re-ID                │
                    └───────────────────────────────────────────┘
                    ┌───────────────────────────────────────────┐
 VidOR ────────────►│ object trajectories ⇒ tracklet GT for Re-ID│
                    │ longer horizons                            │
                    │ ✗ fewer baselines, sparser annotation      │
                    └───────────────────────────────────────────┘
                    ┌───────────────────────────────────────────┐
 Owned QA set ─────►│ StateReducer intervals                     │
 (~100 pairs,       │ Re-ID stitching across gaps                │
  3–5 videos,       │ IR compiler correctness                    │
  static+moving)    │ END-TO-END answer correctness              │
                    └───────────────────────────────────────────┘
```

Annotate the **expected IR tree**, not only the answer. Then a failure localizes immediately: wrong IR is an NL-layer bug, right IR with a wrong answer is a compiler or data bug. Without this, every end-to-end failure requires manual bisection.

### 9.2 Metrics

| Component | Metric | Definition |
|---|---|---|
| Filter | Kept-frame recall | fraction of AG-annotated relation instances whose frame survived filtering |
| Filter | Compression ratio | `frames_in / frames_kept` |
| Detect/Track | Tracklet purity | fraction of a tracklet's frames belonging to its majority GT identity |
| Detect/Track | Tracklet completeness | fraction of a GT identity's frames captured by its majority tracklet |
| Extraction | Edge precision / recall | predicted vs GT `(subject, relation, object)` triples per frame, id-matched via IoU ≥ 0.5 |
| Extraction | Drop rate, overall and per class | from `run_manifest.json` |
| Re-ID | Cluster purity / NMI | against VidOR trajectory ids |
| Re-ID | Fragmentation ratio | `clusters / GT_identities` (>1 = explosion, <1 = fusion) |
| Reducer | Interval IoU | temporal IoU of predicted vs GT `[valid_from, valid_until)` |
| IR | Tree exact-match | predicted IR == annotated IR (structural equality) |
| End-to-end | Answer accuracy | on the QA set |

Fragmentation ratio is the single number to watch on Re-ID: it distinguishes the two opposite failure modes that a purity score alone conflates.

---

## 10. Repo, Config, Packaging

### 10.1 Layout

```
visiongraph/
├── preprocessing/   filter.py  motion_vectors.py  ego_motion.py  embed.py
├── detect/          detector.py  tracker.py  crops.py
├── extractors/      base.py  openrouter.py  sglang.py  som_renderer.py  prompts.py
├── cluster/         reid.py  quality.py  constrained_cluster.py
├── schema/          types.py  builtin/  scaffold.py
├── reduce/          reducer.py  policies.py
├── graph/           base.py  kuzu_store.py  memgraph_store.py  ddl.py
├── retriever/       ir.py  compiler/{kuzu.py,memgraph.py}  nl.py
├── cache/           keys.py  artifacts.py
├── pipeline.py      VideoPipeline — the public façade
└── report.py        graph.report()

eval/       loaders/{action_genome.py,vidor.py}  metrics.py  qa/qa_set.jsonl
tests/      unit/ (reducer, compiler, cache — no network)  integration/
```

### 10.2 Config

One frozen dataclass per stage, composed into one root object, so every knob has exactly one home and the cache key is a pure function of it:

```python
@dataclass(frozen=True)
class PipelineConfig:
    filter: FilterConfig
    detect: DetectConfig
    extract: ExtractConfig
    reid: ReIDConfig
    schema: SkillSchema
    store: StoreConfig
    cache_dir: Path
    max_drop_rate: float | None = None
```

### 10.3 Public API

```python
from visiongraph import VideoPipeline
from visiongraph.schema.builtin import WAREHOUSE

pipeline = VideoPipeline(schema=WAREHOUSE)         # sensible defaults throughout
graph = pipeline.fit("warehouse_cctv.mp4")

print(graph.report())                              # drop rates, counts, warnings

r = graph.query("Did person_1 drop the bag before leaving?")
print(r.answer, r.ir, r.provenance)
```

Two properties worth preserving: `fit()` requires only a schema, and `report()` is impossible to miss before trusting `query()`.

### 10.4 Deployment

| Mode | Composition | Notes |
|---|---|---|
| **Default** | pip + embedded Kuzu + OpenRouter | No Docker, no GPU. **Network-dependent** |
| **Air-gapped** | SGLang + local weights + Memgraph in Compose | The only air-gap-capable configuration (D14 consequence) |
| **Hybrid** | `VideoDBAdapter` replaces the filter; graph logic stays local | Offloads storage/HLS, keeps proprietary logic in-VPC |

Note the correction relative to v2.0.0: the air-gapped claim is **not** true of the default configuration.

---

## 11. Risks and Open Items

### 11.1 ⚠ VERIFY — HEVC motion-vector export

`AV_CODEC_FLAG2_EXPORT_MVS` is implemented in libavcodec for H.264, MPEG-2/4, and VC-1, but is **believed unimplemented for HEVC**. If confirmed, a large share of modern CCTV yields zero motion vectors and Phase 1 silently degrades.

*Test:* decode HEVC and H.264 samples with `export_mvs`, assert non-empty MV side data. *Blocks:* the D3 filter design. *Fallback:* detect empty side data, warn, decode-and-embed.

### 11.2 The eval set is a prerequisite, not a follow-up

Two decisions (D3's probe threshold, the Phase-2 similarity threshold) are untunable without it, and both fail *silently*. Build M6's harness alongside M1, not after M5.

### 11.3 Object Re-ID is weak

OSNet is a person-Re-ID model. Bag and vehicle embeddings are materially less reliable. v0 applies Re-ID only to configured classes; everything else stays tracklet-scoped. This is a real, disclosed limitation, not a tuning problem.

### 11.4 Air-gap contradiction

D14 makes the default network-dependent. Decide whether air-gap is an actual requirement; if yes, SGLang moves from opt-in to a first-class tested path with its own CI.

### 11.5 Cypher dialect divergence

Kuzu's Cypher subset differs from Memgraph's — notably variable-length paths and list/map handling, which `PathBetween` depends on. Inventory the divergence before writing the second compiler backend, and keep one shared operator conformance suite that both dialects must pass.

### 11.6 Extraction quality is still unmeasured

Every accuracy claim above is currently unfalsifiable. M4's eyeball check and M6's AG numbers are the first evidence. Do not build on top of extraction before M6 reports a number.

### 11.7 Deferred design work

- Attribute extraction ("red jacket") for appearance queries — v1 schema extension.
- `group_by` / aggregation operators in the IR.
- Event log as source of truth with State as a droppable materialized view (D10 runner-up) — likely the right v1 model.
- Deferred-commit online Re-ID (D5 runner-up), required for any streaming mode; needs store-level entity merge.
