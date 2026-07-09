#!/usr/bin/env python
"""Assemble the Phase C comparison tables (markdown) from eval/experiments.jsonl + sweep logs.

experiments.jsonl gives per-run metrics + bootstrap CIs + per-stage latency + per-type/lang
(rows are self-describing via the additive ``knobs`` field). The paired A/B verdicts
(Δ / CI / p / ADOPT-TIE) are only printed by --compare/--ab, so we also parse the sweep
stdout logs for them. Output is a set of markdown tables ready to fold into phase_c_report.md.

Usage: python scripts/build_phase_c_report.py [--log eval/experiments.jsonl]
       [--sweep-log <file> ...] [--out eval/phase_c_report_tables.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = ("recall5", "recall10", "ndcg10", "mrr10")
MLABEL = {"recall5": "R@5", "recall10": "R@10", "ndcg10": "nDCG@10", "mrr10": "MRR@10"}
BASE_MODES = ("bm25", "dense", "sparse", "hybrid", "rerank", "routed")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def cell(row: dict, m: str) -> str:
    ci = (row.get("cis") or {}).get(m)
    if ci:
        return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"
    v = (row.get("metrics") or {}).get(m)
    return f"{v:.3f}" if v is not None else "—"


def lat_cell(row: dict) -> str:
    t = (row.get("latency_ms") or {}).get("total", {})
    return f"{t.get('p50', 0):.0f} / {t.get('p95', 0):.0f}"


def extra_knobs(row: dict) -> dict:
    k = dict(row.get("knobs") or {})
    k.pop("rerank_candidates", None)
    return k


def rc_of(row: dict) -> int | None:
    return (row.get("knobs") or {}).get("rerank_candidates")


def dedup_latest(rows: list[dict]) -> list[dict]:
    """Keep the newest row per (mode-label, config_hash, relevance)."""
    best: dict = {}
    for r in sorted(rows, key=lambda r: r.get("timestamp", "")):
        best[(r.get("mode"), r.get("config_hash"), r.get("relevance_level"))] = r
    return list(best.values())


def metric_table(rows: list[dict], label_fn, probe: dict | None = None) -> list[str]:
    """When ``probe`` (depth -> {p50_ms,p95_ms}) is given, rerank rows show CPU rerank latency
    from the probe instead of the row's latency (GPU-offloaded rows carry tunnel latency, not
    the CPU serving latency the report needs)."""
    if not rows:
        return ["_(pending — no runs logged yet)_", ""]
    hdr = "| config | " + " | ".join(MLABEL[m] + " [95% CI]" for m in METRICS) + " | lat p50/p95 (ms) |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    out = [hdr, sep]
    for r in rows:
        cells = " | ".join(cell(r, m) for m in METRICS)
        lat = lat_cell(r)
        if probe and "rerank" in str(r.get("mode")):
            p = probe.get(rc_of(r))
            if p:
                lat = f"{p['p50_ms']:.0f} / {p['p95_ms']:.0f} (CPU)"
        out.append(f"| {label_fn(r)} | {cells} | {lat} |")
    out.append("")
    return out


def slice_table(rows: list[dict], label_fn, group: str, key: str) -> list[str]:
    """Per-query-type or per-language slice (e.g. cross_lingual, en)."""
    field = "per_query_type" if group == "type" else "per_language"
    hdr = "| config | n | " + " | ".join(MLABEL[m] for m in METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    out = [hdr, sep]
    any_row = False
    for r in rows:
        blk = (r.get(field) or {}).get(key)
        if not blk:
            continue
        any_row = True
        cells = " | ".join(f"{blk.get(m, 0):.3f}" for m in METRICS)
        out.append(f"| {label_fn(r)} | {blk.get('n', '?')} | {cells} |")
    out.append("")
    return out if any_row else ["_(pending)_", ""]


PAIR_HDR = re.compile(r"== Paired A/B:\s*(\S+)\s*\(A\)\s*vs\s*(\S+)\s*\(B\),\s*relevance=(\w+)")
PAIR_ROW = re.compile(r"(\w+)\s+Δ=([-+.\d]+)\s*\[([-+.\d]+),\s*([-+.\d]+)\]\s*p=([\d.]+)\s*→\s*(.+)")


def parse_paired(sweep_logs: list[Path]) -> list[dict]:
    comps: list[dict] = []
    cur = None
    for lg in sweep_logs:
        if not lg.exists():
            continue
        for line in lg.read_text(encoding="utf-8", errors="replace").splitlines():
            h = PAIR_HDR.search(line)
            if h:
                cur = {"a": h.group(1), "b": h.group(2), "relevance": h.group(3), "rows": []}
                comps.append(cur)
                continue
            if cur is not None:
                m = PAIR_ROW.search(line)
                if m:
                    cur["rows"].append({
                        "metric": m.group(1), "delta": m.group(2),
                        "lo": m.group(3), "hi": m.group(4), "p": m.group(5),
                        "verdict": m.group(6).strip(),
                    })
    return comps


def paired_section(comps: list[dict]) -> list[str]:
    if not comps:
        return ["_(no paired A/B blocks found in sweep logs)_", ""]
    out = []
    for c in comps:
        out.append(f"**{c['b']} (B) vs {c['a']} (A)** — relevance={c['relevance']}")
        out.append("")
        out.append("| metric | Δ (B−A) | 95% CI | p | verdict |")
        out.append("|---|---|---|---|---|")
        for r in c["rows"]:
            out.append(f"| {MLABEL.get(r['metric'], r['metric'])} | {r['delta']} | "
                       f"[{r['lo']}, {r['hi']}] | {r['p']} | {r['verdict']} |")
        out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="eval/experiments.jsonl")
    ap.add_argument("--gpu-log", nargs="*", default=[])  # rerank/diversity quality (GPU-offloaded)
    ap.add_argument("--probe", default=None)             # rerank_xval.json: CPU rerank latency/depth
    ap.add_argument("--sweep-log", nargs="*", default=[])
    ap.add_argument("--out", default="eval/phase_c_report_tables.md")
    args = ap.parse_args()

    rows = load_rows(Path(args.log))
    for gl in args.gpu_log:  # GPU rerank rows are the only rerank rows → no collision to dedup away
        rows += load_rows(Path(gl))
    probe = {}
    if args.probe and Path(args.probe).exists():
        probe = {int(k): v for k, v in json.loads(Path(args.probe).read_text())["depths"].items()}
    qd = dedup_latest([r for r in rows if (r.get("backend") or {}).get("kind") == "qdrant"])
    chunk = [r for r in qd if r.get("relevance_level") == "chunk"]
    fake_n = sum(1 for r in rows if (r.get("backend") or {}).get("kind") == "fake")

    def by_mode(m):
        return {r.get("mode"): r for r in chunk}.get(m)

    # --- Core modes: base mode label, no extra knobs, default rc=80 -----------------
    core = []
    seen = {r.get("mode"): r for r in chunk if not extra_knobs(r) and rc_of(r) in (None, 80)}
    for m in BASE_MODES:
        if m in seen:
            core.append(seen[m])

    # --- Rerank-depth: mode 'rerank', only rc varies --------------------------------
    depth = sorted([r for r in chunk if r.get("mode") == "rerank" and not extra_knobs(r)],
                   key=lambda r: rc_of(r) or 0)

    # --- Diversity: any max_per_doc / mmr_lambda knob -------------------------------
    div = [r for r in chunk if {"max_per_doc", "mmr_lambda"} & set(extra_knobs(r))]
    div_base = [seen["rerank"]] if "rerank" in seen else []

    # --- Fusion (dbsf), prefetch, ef/rescore, sparse-weight -------------------------
    fusion = [r for r in chunk if extra_knobs(r).get("fusion") == "dbsf"]
    fusion_base = [seen["hybrid"]] if "hybrid" in seen else []
    prefetch = sorted([r for r in chunk if "prefetch_limit" in extra_knobs(r)],
                      key=lambda r: extra_knobs(r)["prefetch_limit"])
    efr = sorted([r for r in chunk if "hnsw_ef" in extra_knobs(r)],
                 key=lambda r: (extra_knobs(r)["hnsw_ef"], str(extra_knobs(r).get("rescore"))))
    sweight = sorted([r for r in chunk if "sparse_weight" in extra_knobs(r)],
                     key=lambda r: extra_knobs(r)["sparse_weight"])

    def lbl_mode(r):
        return f"`{r.get('mode')}`"

    def lbl_rc(r):
        return f"rerank@{rc_of(r)}"

    def lbl_knob(r):
        k = extra_knobs(r)
        return ", ".join(f"{kk}={vv}" for kk, vv in sorted(k.items())) or f"`{r.get('mode')}` (base)"

    md: list[str] = []
    md += ["## Phase C — retrieval comparison (full 2.45M-chunk index, 103-pair golden set)", ""]
    md += [f"_Chunk-level relevance. Metric cells are mean [95% bootstrap CI]. Latency is total "
           f"per-query p50/p95 ms on CPU. {len(qd)} distinct qdrant runs; {fake_n} fake-backend "
           f"(Part-2 synthetic) rows retained for reference._", ""]

    md += ["### 1. Modes (BM25 floor → dense/sparse/hybrid → +rerank → routed)", ""]
    md += ["_rerank = rerank@80; its latency is CPU-derived (see §3 note), others are measured CPU._", ""]
    md += metric_table(core, lbl_mode, probe=probe)

    md += ["### 2. Cross-lingual slice — the routing question (22 EN pairs)", ""]
    md += ["_per-query-type = cross_lingual:_", ""]
    md += slice_table(core, lbl_mode, "type", "cross_lingual")
    md += ["_per-language = en:_", ""]
    md += slice_table(core, lbl_mode, "lang", "en")

    if probe:
        md += ["### 3. Rerank-depth ablation (quality: GPU pod, fp32; latency: CPU, measured)", ""]
        md += ["_Latency is measured on this box by `scripts/rerank_latency_probe.py` "
               "(OMP_NUM_THREADS=8, length-bucketed reranker, reranker-only — add ~0.3-0.5 s "
               "retrieval+encode for end-to-end)._", ""]
    else:
        md += ["### 3. Rerank-depth ablation (quality: GPU pod, fp32; latency: CPU, derived)", ""]
        md += ["_Latency is derived: rerank@80 measured 40.8 s/query end-to-end on the pre-"
               "length-bucketing CPU reranker (4203 s / 103 q); rerank scales ~linearly with depth, "
               "so shallower depths are ~proportional. Treat as an upper bound._", ""]
    md += metric_table(depth, lbl_rc, probe=probe)

    md += ["### 4. Diversity (max-per-doc / MMR) vs no-diversity rerank@80", ""]
    md += metric_table(div_base + div, lbl_knob, probe=probe)

    md += ["### 5. Fusion — DBSF vs RRF (hybrid)", ""]
    md += metric_table(fusion_base + fusion, lbl_knob)

    md += ["### 6. Prefetch recall-pool depth (hybrid)", ""]
    md += metric_table(prefetch, lbl_knob)

    md += ["### 7. HNSW ef × int8 rescore (dense) — quantization recall/latency knee", ""]
    md += metric_table(efr, lbl_knob)

    md += ["### 8. Cross-lingual sparse-weight sweep (hybrid)", ""]
    md += metric_table(sweight, lbl_knob)
    md += ["_cross_lingual slice:_", ""]
    md += slice_table(sweight, lbl_knob, "type", "cross_lingual")

    md += ["### 9. Paired significance (from --compare / --ab)", ""]
    md += paired_section(parse_paired([Path(p) for p in args.sweep_log]))

    # Ranked candidates for the recommendation (by nDCG@10 lower CI bound, then latency).
    def ndcg_lo(r):
        ci = (r.get("cis") or {}).get("ndcg10") or {}
        return ci.get("lo", (r.get("metrics") or {}).get("ndcg10", 0))
    ranked = sorted(chunk, key=lambda r: (-ndcg_lo(r),
                                          (r.get("latency_ms") or {}).get("total", {}).get("p50", 1e9)))
    print("\n=== Candidates ranked by nDCG@10 lower-CI, then p50 latency ===")
    for r in ranked[:15]:
        kn = f"rc={rc_of(r)}"
        ek = extra_knobs(r)
        if ek:
            kn += "," + ",".join(f"{k}={v}" for k, v in sorted(ek.items()))
        print(f"  {r.get('mode'):<14} {kn:<28} ndcg10={cell(r,'ndcg10')}  r@10={cell(r,'recall10')}  "
              f"lat={lat_cell(r)}ms")

    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}  ({len(core)} core modes, {len(depth)} depth, {len(div)} diversity, "
          f"{len(prefetch)} prefetch, {len(efr)} ef/rescore, {len(sweight)} sparse-weight rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
