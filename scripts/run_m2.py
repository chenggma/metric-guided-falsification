"""M2 driver: 7 arms x 5 seeds, sequential, resumable.

Arms (DESIGN.md amendment 2026-08-14): sparse control, three raw-shaped
metrics, three potential-based-shaped (PBRS) metrics. Identical PPO
hyperparameters and identical step budget everywhere; the reward is the
only manipulated variable.

Sequential by measurement, not preference: four concurrent trainings on the
8 GB pilot machine degraded ~12x from memory contention. Each run is a
subprocess so one crash costs one run, not the batch. Resumable: a run
whose model.zip and eval_summary.json both exist is skipped, so the batch
can be interrupted and relaunched with the same command.

Usage: python scripts/run_m2.py [--seeds 5] [--steps 100000] [--outdir results/m2]
"""

import argparse
import json
import os
import subprocess
import sys
import time

CAL = "results/calibration/normalizers.json"
EVAL_MASTER_SEED = 90001  # held-out scenario stream, shared by every arm

ARMS = [
    ("sparse", "raw"),
    ("inv_ttc", "raw"),
    ("neg_tts_margin", "raw"),
    ("pora", "raw"),
    ("inv_ttc", "pbrs"),
    ("neg_tts_margin", "pbrs"),
    ("pora", "pbrs"),
]


def ts() -> str:
    return time.strftime("%H:%M:%S")


def run(cmd, log_path) -> bool:
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode == 0


def main(a) -> None:
    py = sys.executable
    os.makedirs(a.outdir, exist_ok=True)
    manifest = []
    t0 = time.time()
    total = len(ARMS) * a.seeds
    done = 0

    for arm, shaping in ARMS:
        for seed in range(1, a.seeds + 1):
            name = f"{arm}_{shaping}_s{seed}"
            d = os.path.join(a.outdir, name)
            log = os.path.join(a.outdir, f"{name}.log")
            model = os.path.join(d, "model.zip")
            summary = os.path.join(d, "eval_summary.json")
            done += 1
            tag = f"[{ts()}] ({done}/{total}) {name}"

            if os.path.exists(model) and os.path.exists(summary):
                print(f"{tag} skip (complete)", flush=True)
                manifest.append({"name": name, "arm": arm, "shaping": shaping,
                                 "seed": seed, "status": "skipped"})
                continue

            cal = [] if arm == "sparse" else ["--calibration", CAL]

            if not os.path.exists(model):
                print(f"{tag} train", flush=True)
                ok = run([py, "scripts/train_ppo.py", "--arm", arm,
                          "--shaping", shaping, "--seed", str(seed),
                          "--steps", str(a.steps), "--outdir", d] + cal, log)
                if not ok:
                    print(f"{tag} TRAIN FAILED (see {log})", flush=True)
                    manifest.append({"name": name, "arm": arm, "shaping": shaping,
                                     "seed": seed, "status": "train_failed"})
                    continue

            print(f"{tag} eval", flush=True)
            ok = run([py, "scripts/eval_policy.py", "--model", model,
                      "--arm", arm, "--shaping", shaping,
                      "--episodes", str(a.eval_episodes),
                      "--master-seed", str(EVAL_MASTER_SEED),
                      "--out", os.path.join(d, "eval.jsonl")] + cal, log)
            manifest.append({"name": name, "arm": arm, "shaping": shaping,
                             "seed": seed,
                             "status": "ok" if ok else "eval_failed"})
            if not ok:
                print(f"{tag} EVAL FAILED (see {log})", flush=True)

    # Random-OU floor on the same held-out stream.
    rd = os.path.join(a.outdir, "random")
    if not os.path.exists(os.path.join(rd, "eval_summary.json")):
        os.makedirs(rd, exist_ok=True)
        print(f"[{ts()}] random-OU baseline", flush=True)
        run([py, "scripts/eval_policy.py", "--random", "--arm", "sparse",
             "--episodes", str(a.eval_episodes),
             "--master-seed", str(EVAL_MASTER_SEED),
             "--out", os.path.join(rd, "eval.jsonl")],
            os.path.join(a.outdir, "random.log"))

    with open(os.path.join(a.outdir, "manifest.json"), "w") as f:
        json.dump({"arms": ARMS, "seeds": a.seeds, "steps": a.steps,
                   "eval_master_seed": EVAL_MASTER_SEED,
                   "eval_episodes": a.eval_episodes, "runs": manifest}, f, indent=2)

    hrs = (time.time() - t0) / 3600
    fails = [m["name"] for m in manifest if m["status"].endswith("failed")]
    print(f"\n[{ts()}] M2 batch finished in {hrs:.1f} h; "
          f"{len(fails)} failed" + (f": {fails}" if fails else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--outdir", default="results/m2")
    main(ap.parse_args())
