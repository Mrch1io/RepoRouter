"""Rebuild the saved dataset without generating data or overwriting evidence.

Run from the repository root:
    python -m experiments.routing_data.reproduce_data
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lora.format_to_sharegpt import (
    PHASE_2A_ROUTING_SYSTEM_PROMPT,
    convert_to_sharegpt,
    load_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "experiments" / "routing_data" / "outputs"


def reproduce(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing directory: {output_dir}")

    sources: set[Path] = set()

    def read(path: Path, expected_count: int) -> list[dict]:
        sources.add(path)
        rows = load_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(f"{path}: expected {expected_count} records")
        return rows

    v1 = read(RAW / "ft_qa_500_train.reviewed.jsonl", 497)
    implicit = read(RAW / "ft_qa_100_implicit_search.jsonl", 100)
    concept = read(RAW / "ft_qa_40_concept_no_tool.jsonl", 40)
    v2 = v1 + implicit + concept
    if len({row["sample_id"] for row in v2}) != len(v2):
        raise ValueError("Training sources contain duplicate sample_id values")
    if v2 != read(RAW / "ft_qa_train_v2.jsonl", 637):
        raise ValueError("Merged records differ from the saved v2 dataset; no files written")

    in_domain = read(RAW / "ft_qa_450_test.jsonl", 50)
    outputs = {"ft_qa_train_v2.jsonl": v2}
    for name, raw_rows, no_summary in (
        ("routing_train.json", v2, False),
        ("routing_train_v1_no_summary.json", v1, True),
        ("eval_in_domain.json", in_domain, False),
        ("eval_in_domain_no_summary.json", in_domain, True),
    ):
        rows = convert_to_sharegpt(raw_rows, force_phase_2a_system=True)
        if no_summary:
            for row in rows:
                row["conversations"][0]["value"] = PHASE_2A_ROUTING_SYSTEM_PROMPT
        if rows != read(ROOT / "lora" / name, len(raw_rows)):
            raise ValueError(f"{name}: conversion differs from saved records; no files written")
        outputs[name] = rows

    # Keep the evaluation snapshot: regenerating its repository summary would track today's code.
    for name in ("eval_cross_repo.json", "eval_cross_repo_no_summary.json"):
        outputs[name] = read(ROOT / "lora" / name, 87)

    source_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in outputs.items():
        if name.endswith(".jsonl"):
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        else:
            text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        (output_dir / name).write_text(text, encoding="utf-8")
        print(f"Verified {len(rows):3d} records -> {output_dir / name}")

    report = {
        "comparison": "JSON values and record order; whitespace is not compared",
        "reconstructed": list(outputs)[:5],
        "copied_evaluation_snapshots": list(outputs)[5:],
        "source_sha256": source_hashes,
    }
    (output_dir / "VERIFICATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RAW / "reproduced")
    args = parser.parse_args()
    reproduce(args.out_dir)


if __name__ == "__main__":
    main()
