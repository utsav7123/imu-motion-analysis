from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "analysis_summary.json"
SITE_DATA = ROOT / "site" / "data" / "summary.json"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    best = summary["models"]["models"][0]
    payload = {
        "source_file": summary["source_file"],
        "rows": summary["rows"],
        "sampling_hz": summary["sampling_hz"],
        "window_count": summary["window_count"],
        "activity_count": summary["activity_count"],
        "best_model": best["name"],
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "feature_count": summary["models"]["feature_count"],
        "class_counts": summary["models"]["class_counts"],
        "quality": summary["quality"],
    }
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Site summary written to {SITE_DATA}")


if __name__ == "__main__":
    main()
