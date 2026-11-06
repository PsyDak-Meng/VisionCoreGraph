# VisionCoreGraph

Video in, a queryable temporal knowledge graph out. See `SPEC.md` for the full design (15 architectural
decisions, 6-stage pipeline). This README covers what v0 actually implements.

```python
from core import VideoPipeline
from core.schema.builtin.meva import MEVA

pipeline = VideoPipeline(schema=MEVA)
graph = pipeline.fit("eval/qa/clips/object_transfer.mp4")

print(graph.report())

result = graph.query("Did someone pick up a bag before they exited?")   # see Known issues
print(result.answer, result.ir, result.cypher, result.provenance)
```

Or end to end from the shell, build then watch:

```bash
python scripts/run_demo.py eval/qa/clips/object_transfer.mp4 --schema meva
python scripts/build_viewer.py eval/qa/clips/object_transfer.mp4 run_view.html --schema meva
open run_view.html
```

## Install

```bash
conda create -n visioncoregraph python=3.11 -y
conda activate visioncoregraph
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENROUTER_API_KEY
export OPENROUTER_API_KEY=...   # or `source .env` / python-dotenv in your own entrypoint
```

## Pipeline stages implemented in v0

| Stage | Implementation |
|---|---|
| 1. Filter | Fixed-stride frame sampling (`preprocessing/filter.py`) |
| 2. Detect + Track | **Ensemble** (`detect/ensemble.py`): YOLO for COCO-covered schema classes, OWLv2 for the rest, one ByteTrack per class. `detect_backend="yolo"` / `"open_vocab"` select a single detector |
| 3. VLM Extraction | Set-of-Mark overlay + OpenRouter, Pydantic-validated with bounded retry/drop (`extractors/`) |
| 4. Re-ID | **Skipped** — each track id is its own canonical entity, see below |
| 5. State Reduction | Full edge-case policy table from SPEC §4.6 (`reduce/`) |
| 6. Graph Store | Kuzu, DDL generated from the schema (`graph/`) |
| Retrieval | NL → IR → Cypher, executed against Kuzu (`retriever/`) |

## What's deferred (empty stub files, per SPEC §8's own v0 scope)

- `preprocessing/motion_vectors.py`, `ego_motion.py` — the MV coarse gate and ego-motion compensation.
  Fixed-stride sampling is used instead; wasteful but correct.
- `cluster/*` — GlobalReID. **v0 treats each detector/tracker track id as its own canonical entity.**
  On a long video the same person leaving and re-entering frame will be recorded as two separate
  entities. This is a real limitation, not an oversight — SPEC's own v0 scope defers Re-ID.
- `extractors/sglang.py`, `graph/memgraph_store.py`, `retriever/compiler/memgraph.py`,
  `schema/scaffold.py` — second backends and the schema-authoring CLI.
- `eval/loaders/*` — Action Genome / VidOR dataset loaders. The owned QA harness
  (`eval/qa_harness.py`, `eval/metrics.py`) *is* implemented — see `eval/qa/README.md`.

## Other v0 simplifications worth knowing

- **Two built-in schemas**, selected with `--schema`. `warehouse` (person/bag/door/forklift/pallet)
  was written before there was footage; `meva` is fitted to the clips in `eval/qa/clips` by measured
  detectability, adds `trash can` and `DISCARDS`/`BOARDS`/`ALIGHTS`, and drops forklift and pallet,
  which do not occur in MEVA at all. Prefer `meva` on this footage.
- **Detection is an ensemble** (SPEC D4's runner-up, pulled forward). Measured on the 5 MEVA clips,
  neither detector covers WAREHOUSE alone: YOLO finds `person` in all 5 and cannot see `door` at any
  confidence; OWLv2 finds `door` but missed `person` in all 5. Each schema class is routed to whichever
  detector can see it — see `IMPLEMENTATION.md` §2.1 for the numbers.
- **Confidence scales differ by detector and are reconciled before tracking.** COCO reports ~0.3-0.7
  for a confident box, zero-shot OWLv2 ~0.10-0.19. Each is filtered by its own threshold, then the
  tracker's gate sits deliberately low — leaving supervision's 0.25 default is what made every OWLv2
  detection vanish.
- **One ByteTrack per class.** ByteTrack associates on IoU and is class-agnostic, so a shared tracker
  hands a person's id to the door they stand in front of. Ids are namespaced: `person1`, `door2`.
- **Open-vocab detection is slow on CPU** (a full ViT forward pass per keyframe). Raise
  `FilterConfig.stride` on long clips, or use `detect_backend="yolo"` when the schema's classes are
  COCO-covered.
- **Stage cache** (`cache/`) stores stages 2 (detect/track) and 3 (VLM extraction) as JSON, keyed by a
  chained content hash — these are the two costly stages worth skipping on a re-run with an unchanged
  upstream config. Stage 1 (frame decode) and stage 5 (reduction) aren't cached since they're cheap.
- **Retrieval IR**: 6 of the 8 operators and all 3 combinators from SPEC §6.2 are implemented.
  `Overlaps` and multi-hop `PathBetween` are not, and `Precedes` accepts only `EventAt` operands —
  each raises `UnsupportedQueryError` rather than silently returning an empty/wrong answer.
  `And`/`Or`/`Not` combine their children's booleans in Python, so children cannot share a variable:
  "did the *same* person who picked up the bag then exit?" is not expressible.

## Viewing a run

`graph.report()` counts what was extracted; these show what it actually says. Every bug found in
v0 was found here rather than in the report, which read a 0% drop rate throughout all of them.

```bash
# synced player: state intervals on top, video left, live tracking + a real-time
# ring graph and a force-directed static graph right
python scripts/build_viewer.py eval/qa/clips/object_transfer.mp4 run_view.html --schema meva

# or a flat text timeline: entities, events in order, state intervals
python scripts/dump_graph.py .vgc_cache/graphs/<extract_key>/graph.kuzu
```

Omit `--schema` on either for WAREHOUSE. Stage keys include the schema, so the two runs cache
separately and both can coexist over one video.

## Tests

```bash
pytest tests/unit -v          # reducer, compiler, cache keys, QA-set shape — no network, no API key
pytest tests/integration -v   # generates a synthetic clip end-to-end; skips without OPENROUTER_API_KEY
```

## Evaluation

`eval/qa/clips/` holds 5 short surveillance clips from the public
[MEVA dataset](https://mevadata.org) (`s3://mevadata-public-01/examples/videos/`, no-sign-request),
chosen because their activity labels map onto WAREHOUSE relations: pick-up-object, set-down-object,
object-transfer, enter-through-structure, exit-through-structure. They are gitignored — re-fetch with:

```bash
aws s3 cp --no-sign-request s3://mevadata-public-01/examples/videos/ex054-pick-up-object.mp4 \
  eval/qa/clips/pick_up_object.mp4
```

`eval/qa/qa_set.jsonl` currently holds **placeholder** entries (`expected_answer: null`). See
`eval/qa/README.md` for how to annotate them against real clips, then `python -m eval.qa_harness`.

## Known issues in 0.1.0

Both are in the retrieval path. The ingest path (stages 1-6, `fit()`) is unaffected.

- **`graph.query()` fails on every call.** `IRNode` is a recursive discriminated union, and every
  strict structured-output provider rejects the schema it generates — Gemini returns a 500
  (`"An internal error has occurred"`), GPT-4.1-mini a 400 (`additionalProperties is required`).
  A flat schema on the same model and key succeeds, so this is the schema, not the account.
  Workaround until fixed: send `{"type": "json_object"}` and put the JSON schema in the prompt
  instead. `validate_ir` already rejects out-of-vocabulary names, so no enforcement is lost.
- **IR operands match entity ids, not classes.** `subject`/`object` lower to `s.id = $subject`, but
  an LLM given the schema's `entity_classes` writes `"door"`, not `"door5"`. So
  `Count(EventAt(EXITS, object="door"))` answers **0** against a graph holding 13 exits — silently,
  which is the failure mode SPEC §6.5 is written against. SPEC §6.2 specifies that entity `class`
  values are validated and usable here; `validate_ir` only ever checks `relation`.
