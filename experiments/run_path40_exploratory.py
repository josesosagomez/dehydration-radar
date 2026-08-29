"""Run the post-selection path-40 LOSO versus leaky random-session comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dehyd.eval.path40_exploratory import (
    evaluate_path40,
    load_path40_dataset,
    write_outputs,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path40_exploratory.yaml")
    parser.add_argument(
        "--diagnostic-root",
        default="/ibex/user/sosagojm/dehydration_loso_diagnostic",
        help="root of the completed frozen WST-order diagnostic repository",
    )
    args = parser.parse_args(argv)

    repository = Path(__file__).resolve().parents[1]
    config_path = (repository / args.config).resolve(strict=True)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("path-40 config must contain a YAML mapping")

    dataset = load_path40_dataset(args.diagnostic_root, protocol)
    result, predictions = evaluate_path40(dataset, protocol)
    outputs = write_outputs(
        repository, result, predictions, config_path=config_path
    )

    print("PATH 40 EXPLORATORY COMPARISON COMPLETE")
    for name, evaluation in result["evaluations"].items():
        path = evaluation["path40_ridge"]
        reference = evaluation["train_mean_reference"]
        print(
            f"{name}: subject-balanced MAE={path['subject_balanced_mae_pct']:.6f}% "
            f"(train-mean reference={reference['subject_balanced_mae_pct']:.6f}%), "
            f"RMSE={path['session_rmse_pct']:.6f}%, r={path['pooled_pearson_r']}"
        )
    print(f"summary: {outputs['summary']}")
    print(f"predictions: {outputs['predictions']}")
    print("CAUTION: both scores are post-selection exploratory; the random-session score is also leaky.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

