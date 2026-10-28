"""Prepare the 637-example ablation dataset without changing existing artifacts.

Run from any directory:
    python prepare_data.py --root /path/to/RepoAgent --out-dir /path/to/run/data
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def prepare_data(root: Path, out_dir: Path) -> dict:
    root = root.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {out_dir}")

    sys.path.insert(0, str(root))
    from loRA.format_to_sharegpt import (
        PHASE_2A_ROUTING_SYSTEM_PROMPT,
        PHASE_2A_SYSTEM,
    )

    names = (
        "routing_train.json",
        "routing_train_v1_no_summary.json",
        "eval_in_domain_no_summary.json",
        "eval_cross_repo_no_summary.json",
        "dataset_info.json",
    )
    inputs = {name: (root / "loRA" / name).read_bytes() for name in names}
    records = {name: json.loads(content) for name, content in inputs.items()}
    counts = dict(zip(names[:4], (637, 497, 50, 87)))
    for name, expected_count in counts.items():
        rows = records[name]
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} records")
        expected_system = PHASE_2A_SYSTEM if name == "routing_train.json" else PHASE_2A_ROUTING_SYSTEM_PROMPT
        for index, row in enumerate(rows):
            turns = row.get("conversations", [])
            if (
                len(turns) != 3
                or [turn.get("from") for turn in turns] != ["system", "user", "assistant"]
                or any(not isinstance(turn.get("value"), str) for turn in turns)
                or turns[0]["value"] != expected_system
            ):
                raise ValueError(f"{name}: invalid conversation or unexpected system prompt at row {index}")

    training = copy.deepcopy(records["routing_train.json"])
    for row in training:
        row["conversations"][0]["value"] = PHASE_2A_ROUTING_SYSTEM_PROMPT
    if training[:497] != records["routing_train_v1_no_summary.json"]:
        raise ValueError("The first 497 ablation records differ from the saved v1 dataset")

    # Restore only the prompt in a copy to verify every other field and row order.
    restored = copy.deepcopy(training)
    for row in restored:
        row["conversations"][0]["value"] = PHASE_2A_SYSTEM
    if restored != records["routing_train.json"]:
        raise ValueError("Ablation changed content or order outside the system prompt")

    dataset = records["dataset_info.json"].get("repoagent_phase2a_routing", {})
    if dataset != {
        "file_name": "routing_train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }:
        raise ValueError("dataset_info.json: unexpected repoagent_phase2a_routing mapping")

    outputs = {
        "routing_train.json": (json.dumps(training, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        "eval_in_domain_no_summary.json": inputs["eval_in_domain_no_summary.json"],
        "eval_cross_repo_no_summary.json": inputs["eval_cross_repo_no_summary.json"],
        "dataset_info.json": inputs["dataset_info.json"],
    }
    validation = {
        "input_sha256": {
            f"loRA/{name}": hashlib.sha256(content).hexdigest()
            for name, content in inputs.items()
        },
        "output_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()
        },
        "counts": {"training": 637, "v1_prefix": 497, "supplement": 140, "in_domain": 50, "cross_repo": 87},
        "checks": {
            "only_training_system_prompt_changed": True,
            "all_other_fields_and_record_order_preserved": True,
            "first_497_records_equal_v1": True,
            "all_prompts_equal_v1_no_summary": True,
            "evaluation_and_dataset_mapping_bytes_unchanged": True,
        },
    }
    validation["input_sha256"]["loRA/format_to_sharegpt.py"] = hashlib.sha256(
        (root / "loRA" / "format_to_sharegpt.py").read_bytes()
    ).hexdigest()
    outputs["data_validation.json"] = (json.dumps(validation, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    out_dir.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        with (out_dir / name).open("xb") as stream:
            stream.write(content)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    result = prepare_data(args.root, args.out_dir)
    print(f"Verified {result['counts']['training']} training examples -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
