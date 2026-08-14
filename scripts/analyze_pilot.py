"""Summarize pilot training logs: does any arm learn to falsify?

Reads train_episodes.jsonl per arm and reports, per arm:
outcome distribution, first non-trivial failure (episode / cumulative env
step), and non-trivial failures per 10k steps - overall and in the last
quarter of training (a crude "after learning" read; the real comparison is
the held-out eval on identical seeds).

Usage: python scripts/analyze_pilot.py results/pilot
"""

import collections
import json
import os
import sys

NON_TRIVIAL = {"ego_attacker_crash", "ego_third_party_crash"}
TRIVIAL = "ego_attacker_crash_trivial"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize(records):
    outcomes = collections.Counter(r["outcome"] for r in records)
    steps = records[-1]["cumulative_steps"] if records else 0
    first_nt = next(
        (
            {"episode": i + 1, "cumulative_steps": r["cumulative_steps"]}
            for i, r in enumerate(records)
            if r["outcome"] in NON_TRIVIAL
        ),
        None,
    )
    nt_total = sum(outcomes[o] for o in NON_TRIVIAL)

    q0 = 3 * steps // 4
    late = [r for r in records if r["cumulative_steps"] > q0]
    late_steps = steps - q0
    late_nt = sum(1 for r in late if r["outcome"] in NON_TRIVIAL)

    return {
        "episodes": len(records),
        "env_steps": steps,
        "outcomes": dict(outcomes),
        "first_non_trivial": first_nt,
        "non_trivial_per_10k": 1e4 * nt_total / steps if steps else 0.0,
        "non_trivial_per_10k_last_quarter": (
            1e4 * late_nt / late_steps if late_steps else 0.0
        ),
        "mean_episode_len": steps / len(records) if records else 0.0,
    }


def main(root):
    out = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "train_episodes.jsonl")
        if not os.path.isfile(path):
            continue
        out[name] = summarize(load(path))

    for name, s in out.items():
        print(f"\n== {name}: {s['episodes']} eps, {s['env_steps']} steps, "
              f"mean len {s['mean_episode_len']:.1f}")
        for o, n in sorted(s["outcomes"].items(), key=lambda kv: -kv[1]):
            print(f"   {o:<28} {n:>6}  ({100 * n / s['episodes']:.1f}%)")
        print(f"   first non-trivial: {s['first_non_trivial']}")
        print(f"   non-trivial /10k steps: {s['non_trivial_per_10k']:.2f} "
              f"(last quarter: {s['non_trivial_per_10k_last_quarter']:.2f})")

    with open(os.path.join(root, "pilot_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(root, 'pilot_summary.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pilot")
