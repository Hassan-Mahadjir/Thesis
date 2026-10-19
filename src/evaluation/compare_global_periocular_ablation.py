from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.global_local.common import resolve_path


DEFAULT_VARIANTS = [
    "global_pair",
    "local_pair",
    "global_local_pair",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    cfg = config["global_periocular"]

    # Robust fallback for older config copies.
    variants = cfg.get("diagnostic_variants", DEFAULT_VARIANTS)

    report_root = (
        resolve_path(config["output"]["training_reports_dir"])
        / "ablation"
    )

    rows = []

    for variant in variants:
        summary_path = report_root / variant / "tuning_summary.json"
        history_path = report_root / variant / "tuning_history.csv"

        if not summary_path.exists():
            print(f"Missing summary for {variant}: {summary_path}")
            continue

        if not history_path.exists():
            print(f"Missing history for {variant}: {history_path}")
            continue

        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        history = pd.read_csv(history_path)

        best_epoch_value = summary.get("best_epoch")
        if best_epoch_value is None:
            raise ValueError(
                f"{summary_path} has no best_epoch."
            )

        best_epoch = int(best_epoch_value)
        best_rows = history.loc[history["epoch"] == best_epoch]

        if best_rows.empty:
            raise ValueError(
                f"Best epoch {best_epoch} is not present in {history_path}."
            )

        best_row = best_rows.iloc[0]

        row = {
            "variant": variant,
            "best_epoch": best_epoch,
            "weighted_eer": float(best_row["validation_objective"]),
        }

        for protocol_name in cfg["validation_protocols"].keys():
            column = f"eer::{protocol_name}"
            if column in best_row.index:
                row[protocol_name] = float(best_row[column])

        rows.append(row)

    if not rows:
        raise SystemExit("No complete ablation results were found.")

    result = pd.DataFrame(rows).sort_values(
        "weighted_eer",
        ascending=True,
    )

    display = result.copy()

    for column in display.columns:
        if column not in {"variant", "best_epoch"}:
            display[column] = (
                display[column] * 100.0
            ).map(lambda value: f"{value:.2f}%")

    print()
    print("DEVELOPMENT-ONLY ARCHITECTURAL ABLATION")
    print("=" * 100)
    print(display.to_string(index=False))
    print("=" * 100)

    output_path = report_root / "ablation_comparison.csv"
    result.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")

    if set(DEFAULT_VARIANTS).issubset(set(result["variant"])):
        lookup = result.set_index("variant")

        gp = float(lookup.loc["global_pair", "weighted_eer"])
        lp = float(lookup.loc["local_pair", "weighted_eer"])
        gl = float(lookup.loc["global_local_pair", "weighted_eer"])

        print("\nQuick interpretation")
        print("--------------------")
        print(f"global_pair       : {gp * 100:.2f}%")
        print(f"local_pair        : {lp * 100:.2f}%")
        print(f"global_local_pair : {gl * 100:.2f}%")

        if gl < gp:
            print(
                f"- Adding periocular information improves the learned "
                f"global pair head by {(gp - gl) * 100:.2f} percentage points."
            )
        else:
            print(
                f"- Adding periocular information worsens the learned "
                f"global pair head by {(gl - gp) * 100:.2f} percentage points."
            )

        if lp > gp:
            print(
                "- Periocular-only is substantially weaker than global-only; "
                "use it as complementary evidence, not as a replacement."
            )

        print(
            "- Do not run these diagnostic variants on FINAL_TEST."
        )


if __name__ == "__main__":
    main()
