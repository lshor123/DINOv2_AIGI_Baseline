"""Validate complete benchmark outputs and write simple Markdown/JSON tables."""
import csv
import json
import sys
from pathlib import Path

RUNS = ["D1_dynamic_local", "C1_linear_continue", "A1_dynamic_no_local", "C2_residual_mlp"]
GENIMAGE = ["adm_imagenet", "biggan_imagenet", "glide_imagenet", "midjourney_imagenet",
            "sdv4_imagenet", "sdv5_imagenet", "vqdm_imagenet", "wukong_imagenet"]
G5_ROOT = Path("/root/autodl-tmp/outputs/dinov2_vitl14_baseline/linear_probe_zoom_crop_matrix/"
               "G5_headlr_3em4_train_only_deterministic_val")


def read_eval(root, benchmark):
    files = list((root / (benchmark + "_eval") / "results").glob("eval_*.csv"))
    if not files:
        raise FileNotFoundError(f"No completed {benchmark} CSV for {root}")
    source = max(files, key=lambda path: path.stat().st_mtime)
    with source.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = next(i for i, row in enumerate(rows)
                  if row[:4] == ["Dataset", "Accuracy", "AUC", "AP"])
    metrics = {row[0]: [float(v) for v in row[1:4]] for row in rows[header + 1:] if row}
    expected = set(GENIMAGE if benchmark == "genimage" else ["."])
    if set(metrics) != expected | {"MEAN"}:
        raise ValueError(f"Incomplete dataset metrics in {source}: {list(metrics)}")
    for values in metrics.values():
        if len(values) != 3 or any(not 0 <= value <= 100 for value in values):
            raise ValueError(f"Invalid metric values in {source}")
    means = [sum(metrics[name][i] for name in expected) / len(expected) for i in range(3)]
    if any(abs(a - b) > 1e-8 for a, b in zip(means, metrics["MEAN"])):
        raise ValueError(f"Inconsistent macro average in {source}")
    return {"source": str(source), "metrics": metrics}


def summarize(output):
    records = {}
    # Historical G5 is useful context, but the continued linear head is the
    # matched additional-training control (all four new runs use FP32 heads).
    records["G5_historical"] = {b: read_eval(G5_ROOT, b) for b in ("genimage", "chameleon")}
    for run in RUNS:
        root = output / run
        complete = json.loads((root / "training_complete.json").read_text())
        best = json.loads((root / "best_checkpoint.json").read_text())
        if not Path(best["path"]).is_file():
            raise FileNotFoundError(best["path"])
        records[run] = {"training": complete, "best_checkpoint": best,
                        **{b: read_eval(root, b) for b in ("genimage", "chameleon")}}
    lines = ["# Dynamic-head experiment results", "", "All metrics are percentages. Each cell: Acc / AUC / AP.",
             "", "G5_historical is the previous result. New runs warm-start its best checkpoint, reset the optimizer,",
             "and use the same FP32-head / AMP-backbone precision policy. Best epoch means additional epoch.", ""]
    for benchmark in ("genimage", "chameleon"):
        lines.extend(["## " + benchmark, "", "| Dataset | " + " | ".join(records) + " |",
                      "|---|" + "---|" * len(records)])
        subsets = GENIMAGE + ["MEAN"] if benchmark == "genimage" else ["."]
        for subset in subsets:
            cells = [" / ".join(f"{v:.2f}" for v in item[benchmark]["metrics"][subset]) for item in records.values()]
            lines.append("| " + subset + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Additional training", "", "| Run | Completed epochs | Best epoch |", "|---|---:|---:|"])
    for run in RUNS:
        item = records[run]
        lines.append(f"| {run} | {item['training']['completed_epochs']} | {item['best_checkpoint']['epoch']} |")
    lines.extend(["", "## D1 minus controls", "", "Differences are percentage points (Acc / AUC / AP).", ""])
    deltas = {}
    for benchmark in ("genimage", "chameleon"):
        deltas[benchmark] = {}
        values = records[RUNS[0]][benchmark]["metrics"]["MEAN"]
        for reference in list(records)[0:1] + RUNS[1:]:
            baseline = records[reference][benchmark]["metrics"]["MEAN"]
            diff = [a - b for a, b in zip(values, baseline)]
            deltas[benchmark][reference] = diff
            lines.append(f"- {benchmark}, D1 - {reference}: " + " / ".join(f"{v:+.2f}" for v in diff))
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "RESULTS.json").write_text(json.dumps({"runs": records, "deltas": deltas}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
