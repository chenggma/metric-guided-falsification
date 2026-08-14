"""M2 driver: 7 arms x 5 seeds, sequential, resumable.

Arms (DESIGN.md amendment 2026-08-14): sparse control, three raw-shaped
metrics, three potential-based-shaped (PBRS) metrics. Identical PPO
hyperparameters and identical step budget everywhere; the reward is the
only manipulated variable.

Ordering is SEED-MAJOR: seed 1 of every arm, then seed 2 of every arm, and
so on. Arm-major ordering would leave the batch uninterpretable until the
last arm finished; seed-major means that after one pass there is already a
complete (if underpowered) cross-arm comparison, and each further pass adds
statistical power to a table that already exists. An interrupted batch is
therefore still analyzable.

`--jobs` sets concurrency, and it is set from measurement on this machine
(4P+4E, 8 GB M3), 2026-08-14:

    1 process   85 steps/s
    2 processes 83 + 67 = 150 steps/s aggregate (1.76x), 45% memory free

An earlier version of this file asserted that concurrency "degraded ~12x
from memory contention." That number was never measured - it was an
assumption written as a finding, and the measurement above refutes it.
jobs=2 is the default-safe choice: it saturates the four performance cores
(2 processes x OMP_NUM_THREADS=2) while leaving enough headroom that an
unattended multi-hour batch cannot fall into a swap spiral, which is the
real risk on 8 GB and the likely origin of the discarded intuition.

Each run is a subprocess so one crash costs one run, not the batch.
Resumable: a run whose model.zip and eval_summary.json both exist is
skipped, so the batch can be interrupted and relaunched with the same
command.

Usage: python scripts/run_m2.py [--seeds 5] [--steps 100000] [--jobs 1]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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


def one_run(py, a, arm, shaping, seed, index, total):
    """Train (if needed) then evaluate one (arm, shaping, seed). Returns a
    manifest entry. Safe to call from a worker thread: it only shells out."""
    name = f"{arm}_{shaping}_s{seed}"
    d = os.path.join(a.outdir, name)
    log = os.path.join(a.outdir, f"{name}.log")
    model = os.path.join(d, "model.zip")
    summary = os.path.join(d, "eval_summary.json")
    tag = f"[{ts()}] ({index}/{total}) {name}"
    entry = {"name": name, "arm": arm, "shaping": shaping, "seed": seed}

    if os.path.exists(model) and os.path.exists(summary):
        print(f"{tag} skip (complete)", flush=True)
        return {**entry, "status": "skipped"}

    cal = [] if arm == "sparse" else ["--calibration", CAL]

    if not os.path.exists(model):
        print(f"{tag} train", flush=True)
        if not run([py, "scripts/train_ppo.py", "--arm", arm,
                    "--shaping", shaping, "--seed", str(seed),
                    "--steps", str(a.steps), "--outdir", d] + cal, log):
            print(f"{tag} TRAIN FAILED (see {log})", flush=True)
            return {**entry, "status": "train_failed"}

    print(f"{tag} eval", flush=True)
    ok = run([py, "scripts/eval_policy.py", "--model", model,
              "--arm", arm, "--shaping", shaping,
              "--episodes", str(a.eval_episodes),
              "--master-seed", str(EVAL_MASTER_SEED),
              "--out", os.path.join(d, "eval.jsonl")] + cal, log)
    if not ok:
        print(f"{tag} EVAL FAILED (see {log})", flush=True)
    return {**entry, "status": "ok" if ok else "eval_failed"}


def main(a) -> None:
    py = sys.executable
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()

    # Seed-major: one full cross-arm pass per seed, so partial batches are
    # already analyzable and each pass only adds power.
    work = [
        (arm, shaping, seed)
        for seed in range(1, a.seeds + 1)
        for arm, shaping in ARMS
    ]
    total = len(work)

    if a.jobs > 1:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures = [
                pool.submit(one_run, py, a, arm, shaping, seed, i + 1, total)
                for i, (arm, shaping, seed) in enumerate(work)
            ]
            manifest = [f.result() for f in futures]
    else:
        manifest = [
            one_run(py, a, arm, shaping, seed, i + 1, total)
            for i, (arm, shaping, seed) in enumerate(work)
        ]

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
                   "jobs": a.jobs, "order": "seed-major",
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
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent runs; set from measurement, not intuition")
    ap.add_argument("--outdir", default="results/m2")
    main(ap.parse_args())
