"""Aggregate L2 judge verdicts → faithfulness / correctness numbers (see ``eval/judge_eval.py``).

Reads one or more verdicts JSONL files (pass several to form a **panel** — majority vote per
axis blunts single-judge bias), aggregates, prints a report, and writes a results JSON.

    python -m eval.score_judge --verdicts .state/answer_eval/verdicts_run1.jsonl
    python -m eval.score_judge --verdicts v_judgeA.jsonl v_judgeB.jsonl v_judgeC.jsonl  # panel-of-3
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .judge_eval import aggregate_verdicts, breakdown_verdicts, load_verdicts, merge_panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Score L2 judge verdicts (faithfulness/correctness).")
    ap.add_argument("--verdicts", nargs="+", required=True,
                    help="one or more verdicts JSONL (>1 = panel, majority vote per axis)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panels = [load_verdicts(Path(p)) for p in args.verdicts]
    merged = merge_panel(panels) if len(panels) > 1 else panels[0]
    agg = aggregate_verdicts(merged)
    by_type = breakdown_verdicts(merged, "query_type")

    n_judges = len(panels)
    print(f"\n== L2 answer faithfulness ({'panel-of-' + str(n_judges) if n_judges > 1 else 'single judge'}, "
          f"n={agg.n}) ==")
    print(f"  faithfulness (grounded, THE #1 metric): {agg.faithfulness:.3f}")
    print(f"  correctness  (matches gold answer):     {agg.correctness:.3f}")
    print(f"  completeness (context sufficient):      {agg.completeness:.3f}")
    print(f"  abstain rate (judged should-refuse):    {agg.abstain_rate:.3f}")
    print("\n  by query_type:")
    for t, a in by_type.items():
        print(f"    {t:<18} n={a['n']:<3} faithful={a['faithfulness']:.3f} correct={a['correctness']:.3f}")

    out_path = Path(args.out) if args.out else Path(args.verdicts[0]).with_suffix(".scored.json")
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_judges": n_judges,
        "verdict_files": list(args.verdicts),
        "overall": agg.__dict__,
        "per_query_type": by_type,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
