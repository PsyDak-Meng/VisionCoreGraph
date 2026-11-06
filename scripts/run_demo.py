"""Quickstart runner: python scripts/run_demo.py <video_path> ["question"] [--schema NAME]"""

import sys

from dotenv import load_dotenv

from core.pipeline import VideoPipeline
from core.schema.builtin.meva import MEVA
from core.schema.builtin.warehouse import WAREHOUSE

SCHEMAS = {"warehouse": WAREHOUSE, "meva": MEVA}

load_dotenv()


def main() -> None:
    args = sys.argv[1:]
    schema = WAREHOUSE
    if "--schema" in args:
        i = args.index("--schema")
        schema = SCHEMAS[args[i + 1]]
        del args[i : i + 2]
    video_path = args[0]
    question = args[1] if len(args) > 1 else None

    pipeline = VideoPipeline(schema=schema)
    graph = pipeline.fit(video_path)

    print(graph.report())

    if question:
        result = graph.query(question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")
        print(f"IR: {result.ir}")
        print(f"Cypher: {result.cypher}")
        print(f"Provenance (frames): {result.provenance}")


if __name__ == "__main__":
    main()
