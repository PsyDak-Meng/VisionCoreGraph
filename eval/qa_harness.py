"""Runs the owned QA set (SPEC §9.1): the only ground truth that exercises the reducer,
the IR compiler, and end-to-end answer correctness together. Annotating the expected IR
tree (not just the answer) is deliberate — a wrong IR is an NL-layer bug, a right IR with
a wrong answer is a compiler or data bug (§9.1).
"""

from __future__ import annotations

import json
from pathlib import Path

from core.pipeline import PipelineConfig, VideoPipeline
from core.schema.builtin.warehouse import WAREHOUSE
from eval.metrics import answer_accuracy, ir_exact_match

QA_SET_PATH = Path(__file__).parent / "qa" / "qa_set.jsonl"


def load_qa_set(path: Path = QA_SET_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(qa_set_path: Path = QA_SET_PATH, cache_dir: Path = Path(".vgc_cache")) -> dict:
    pairs = load_qa_set(qa_set_path)
    graphs: dict[Path, object] = {}
    results = []
    skipped = []

    for pair in pairs:
        video = (qa_set_path.parent / pair["video"]).resolve()
        if not video.exists():
            skipped.append(pair["id"])
            continue
        if video not in graphs:
            config = PipelineConfig(schema=WAREHOUSE, cache_dir=cache_dir)
            graphs[video] = VideoPipeline(schema=WAREHOUSE, config=config).fit(video)
        graph = graphs[video]

        result = graph.query(pair["question"])
        results.append({
            "id": pair["id"],
            "ir_match": ir_exact_match(result.ir, pair["expected_ir"]),
            "actual_answer": result.answer,
            "expected_answer": pair["expected_answer"],
        })

    return {
        "n_total": len(pairs),
        "n_skipped_missing_video": len(skipped),
        "skipped_ids": skipped,
        "ir_exact_match_rate": sum(r["ir_match"] for r in results) / len(results) if results else 0.0,
        "answer_accuracy": answer_accuracy(results),
        "results": results,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
