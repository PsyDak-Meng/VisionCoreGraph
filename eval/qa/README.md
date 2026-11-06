# Owned QA set (SPEC §9.1)

The only ground truth that exercises the reducer, IR compiler, and end-to-end answer
correctness together — public datasets (Action Genome, VidOR) encode per-frame states, not
events, so they can't validate the reducer, and their clips are too short for Re-ID.

## Format (`qa_set.jsonl`, one JSON object per line)

```json
{
  "id": "qa_001",
  "video": "clips/pickup_handoff.mp4",
  "question": "Did person t1 possess the bag at t=5 seconds?",
  "expected_ir": {"op": "holds_at", "relation": "POSSESSES", "subject": "t1", "object": "t2", "t": 5.0},
  "expected_answer": true,
  "notes": "optional context"
}
```

`expected_ir` is annotated, not just `expected_answer` — a wrong IR is an NL-layer bug, a right
IR with a wrong answer is a compiler or data bug. Without both, every failure needs manual
bisection (§9.1).

## Adding a clip

1. Drop a short clip (10-30s to start) into `clips/` — aim for 3-5 clips total, a mix of static
   and moving camera per SPEC §9.1. `clips/` is gitignored by default; these are your own footage.
2. Run it once: `python scripts/run_demo.py eval/qa/clips/your_clip.mp4` and read `graph.report()`
   to see which entities the detector actually tracked (their `t<N>` ids) — you need the real ids
   to write `subject`/`object` correctly, not guesses.
3. Watch the clip yourself, write the question, the expected IR tree using those ids, and the
   ground-truth answer by hand.
4. Append a line to `qa_set.jsonl`.

## Current entries

The 5 entries checked in now are **placeholders** — `expected_answer: null` and TODO notes,
referencing clips that don't exist yet. They exist to fix the QA-pair shape and exercise every
implemented IR operator (`HoldsAt`, `HoldsDuring`, `EventAt`, `Precedes`, `Count`, `And`). Replace
them with real clips and real ground truth before trusting any number out of the harness.

## Running

```bash
python -m eval.qa_harness
```

Entries whose `video` file doesn't exist are skipped and counted, not treated as failures —
this lets the harness run cleanly before every clip is authored.
