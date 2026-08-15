"""M2 analysis: efficiency, farming diagnostics, diversity, cross-metric transfer.

Unit of replication is the SEED, so every headline number is a median over
seeds with a percentile bootstrap CI over seeds, and every comparison is an
exact Mann-Whitney over the per-seed values (falsify/stats.py). Only the
contrasts registered in DESIGN.md are tested, and they carry Holm-adjusted
p-values; the 5-vs-5 design floors two-sided p at 2/252 = 0.0079.

Usage: python scripts/analyze_m2.py [results/m2] [--calibration ...]
"""

import argparse
import collections
import json
import math
import os

import numpy as np

from falsify.stats import bootstrap_ci, holm, mannwhitney_exact

NON_TRIVIAL = {"ego_attacker_crash", "ego_third_party_crash"}
# Severity floor, applied here as well as in the reward so that runs logged
# BEFORE the gate existed are scored the same way as runs logged after.
# See falsify/fault.py for why 5.0 m/s and how it was chosen.
SEVERITY_MIN_CLOSING = 5.0


def is_failure(rec, min_closing=SEVERITY_MIN_CLOSING):
    """A severe, SUT-attributable failure: correct fault label AND a closing
    speed above the fender-bender floor."""
    if rec["outcome"] not in NON_TRIVIAL:
        return False
    snap = rec.get("crash_snapshot")
    if not snap:
        return False
    return snap["closing_speed"] >= min_closing


METRIC_NAMES = ("inv_ttc", "neg_tts_margin", "pora")

# Deterministic failure-mode grid (no clustering library, no random seed):
# closing speed x |relative heading| x ego speed.
CLOSING_BINS = [5.0, 10.0, 15.0, 20.0]
HEADING_BINS = [0.15, 0.40, 0.80]     # parallel / slight / oblique / broadside
EGO_SPEED_BINS = [10.0, 15.0, 20.0]


def _bin(x, edges):
    return int(np.digitize([x], edges)[0])


def cell(snap):
    return (
        _bin(snap["closing_speed"], CLOSING_BINS),
        _bin(abs(snap["rel_heading_rad"]), HEADING_BINS),
        _bin(snap["ego_speed"], EGO_SPEED_BINS),
    )


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def collect(root):
    """runs[(arm, shaping)][seed] = {eval records, train records, summary}"""
    runs = collections.defaultdict(dict)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        summary_path = os.path.join(d, "eval_summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if name == "random":
            runs[("random-OU", "-")][0] = {
                "summary": summary,
                "eval": load_jsonl(os.path.join(d, "eval.jsonl")),
                "train": [],
            }
            continue
        arm, shaping, seed_tag = name.rsplit("_", 2)
        runs[(arm, shaping)][int(seed_tag[1:])] = {
            "summary": summary,
            "eval": load_jsonl(os.path.join(d, "eval.jsonl")),
            "train": load_jsonl(os.path.join(d, "train_episodes.jsonl")),
        }
    return runs


def per_seed_nontrivial(runs_for_arm):
    return [
        sum(1 for r in v["eval"] if is_failure(r))
        for _, v in sorted(runs_for_arm.items())
    ]


def farming(runs_for_arm):
    """(timeout share, mean return on timeout episodes) over late training."""
    shares, returns = [], []
    for _, v in sorted(runs_for_arm.items()):
        recs = v["train"]
        if not recs:
            continue
        late = recs[3 * len(recs) // 4:]
        touts = [r for r in late if r["outcome"] == "timeout"]
        shares.append(len(touts) / len(late))
        if touts:
            returns.append(sum(r["return"] for r in touts) / len(touts))
    return shares, returns


def main(a):
    runs = collect(a.root)
    if not runs:
        print(f"no completed runs under {a.root}")
        return

    report = {"efficiency": {}, "farming": {}, "diversity": {}, "transfer": {}}

    print("=" * 74)
    print("EFFICIENCY - non-trivial failures per 200 held-out episodes (shared seeds)")
    print("=" * 74)
    print(f"{'arm':>22} {'n':>3}  {'per-seed':<20} {'median':>7}  {'95% CI':>13}")
    per_seed = {}
    for key in sorted(runs, key=lambda k: (k[1], k[0])):
        vals = per_seed_nontrivial(runs[key])
        per_seed[key] = vals
        est, lo, hi = bootstrap_ci(vals)
        label = f"{key[0]}/{key[1]}"
        print(f"{label:>22} {len(vals):>3}  {str(vals):<20} {est:>7.1f}  [{lo:>4.1f},{hi:>5.1f}]")
        report["efficiency"][label] = {"per_seed": vals, "median": est,
                                       "ci95": [lo, hi]}

    # Pre-registered contrasts only (DESIGN.md amendment): sparse vs each arm,
    # and raw vs pbrs within each metric.
    contrasts = []
    sparse_key = ("sparse", "raw")
    if sparse_key in per_seed:
        for key in per_seed:
            if key != sparse_key and key[0] != "random-OU" and len(per_seed[key]) > 1:
                contrasts.append((f"sparse vs {key[0]}/{key[1]}",
                                  per_seed[sparse_key], per_seed[key]))
    for m in METRIC_NAMES:
        if (
            len(per_seed.get((m, "raw"), [])) > 1
            and len(per_seed.get((m, "pbrs"), [])) > 1
        ):
            contrasts.append((f"{m}: raw vs pbrs",
                              per_seed[(m, "raw")], per_seed[(m, "pbrs")]))

    if contrasts:
        print("\n" + "=" * 74)
        print("PRE-REGISTERED CONTRASTS - exact Mann-Whitney, Holm-adjusted")
        print("=" * 74)
        raw_ps = []
        for label, x, y in contrasts:
            try:
                u, p = mannwhitney_exact(x, y)
            except ValueError:
                continue
            raw_ps.append((label, p))
        sizes = {len(v) for v in per_seed.values() if len(v) > 1}
        floor = 2 / math.comb(10, 5) if sizes == {5} else None
        for label, p, adj, rej in holm(raw_ps):
            print(f"  {label:<34} p={p:.4f}  Holm={adj:.4f}  {'REJECT' if rej else '-'}")
            report.setdefault("contrasts", {})[label] = {"p": p, "holm": adj,
                                                         "reject": rej}
        if floor:
            print(f"  (design floor: smallest attainable two-sided p = {floor:.4f})")

    print("\n" + "=" * 74)
    print("FARMING DIAGNOSTICS - late training (last quarter)")
    print("=" * 74)
    print(f"{'arm':>22}  {'timeout share':>14}  {'timeout-ep return':>18}")
    for key in sorted(runs, key=lambda k: (k[1], k[0])):
        if key[0] == "random-OU":
            continue
        shares, returns = farming(runs[key])
        if not shares:
            continue
        s = f"{100 * float(np.median(shares)):.1f}%"
        r = f"{float(np.median(returns)):.1f}" if returns else "-"
        label = f"{key[0]}/{key[1]}"
        print(f"{label:>22}  {s:>14}  {r:>18}")
        report["farming"][label] = {"timeout_share_median": float(np.median(shares)),
                                    "timeout_return_median": (
                                        float(np.median(returns)) if returns else None)}

    print("\n" + "=" * 74)
    print("DIVERSITY - distinct failure-mode cells (closing x heading x ego speed)")
    print("=" * 74)
    for key in sorted(runs, key=lambda k: (k[1], k[0])):
        cells_per_seed, union = [], set()
        for _, v in sorted(runs[key].items()):
            cs = {cell(r["crash_snapshot"]) for r in v["eval"] if is_failure(r)}
            cells_per_seed.append(len(cs))
            union |= cs
        if not any(cells_per_seed):
            continue
        est, lo, hi = bootstrap_ci(cells_per_seed)
        label = f"{key[0]}/{key[1]}"
        print(f"{label:>22}  per-seed {str(cells_per_seed):<18} "
              f"median {est:.1f} [{lo:.1f},{hi:.1f}]  union {len(union)}")
        report["diversity"][label] = {"per_seed_cells": cells_per_seed,
                                      "median": est, "union": len(union),
                                      "cells": sorted(map(list, union))}

    print("\n" + "=" * 74)
    print("CROSS-METRIC TRANSFER - median episode-max metric on THIS arm's failures")
    print("=" * 74)
    print(f"{'found by':>22}  " + "  ".join(f"{m:>16}" for m in METRIC_NAMES))
    for key in sorted(runs, key=lambda k: (k[1], k[0])):
        vals = {m: [] for m in METRIC_NAMES}
        for _, v in sorted(runs[key].items()):
            for r in v["eval"]:
                if is_failure(r):
                    for m in METRIC_NAMES:
                        if m in r.get("max_metrics", {}):
                            vals[m].append(r["max_metrics"][m])
        if not any(vals.values()):
            continue
        label = f"{key[0]}/{key[1]}"
        cells = "  ".join(
            f"{np.median(vals[m]):>16.3g}" if vals[m] else f"{'-':>16}"
            for m in METRIC_NAMES
        )
        print(f"{label:>22}  {cells}")
        report["transfer"][label] = {m: (float(np.median(vals[m])) if vals[m] else None)
                                     for m in METRIC_NAMES}

    out = os.path.join(a.root, "m2_summary.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/m2")
    main(ap.parse_args())
