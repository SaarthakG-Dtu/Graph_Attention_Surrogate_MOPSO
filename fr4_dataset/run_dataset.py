"""
run_dataset.py — CST sweep of the FR-4 16x16 RDN dataset, one design at a time.

Interruption-safe by construction:
  * every design is saved as its own file, samples_<tag>/sample_XXXXXX.npz,
    the moment its simulation finishes;
  * a file is written to a temporary name and then renamed, so a crash or a
    power cut can never leave a half-written sample behind;
  * on start, finished samples are skipped (a damaged file is re-simulated);
  * a worker claims a design with a lock file before simulating it, so several
    workers (on one PC or on several PCs sharing this folder) never duplicate
    work; locks of crashed workers expire after --stale_hours;
  * a failed design is logged and retried on the next run (up to --max_tries).

Run (one worker per PC is recommended; split the index range across PCs):
    python run_dataset.py                          # indices 0 .. N_SIMULATE-1
    python run_dataset.py --start 0 --end 15000    # PC 1
    python run_dataset.py --start 15000 --end 30000 --worker pc2
Stop at any time with Ctrl+C (or by closing the window); just run it again.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

import fr4_config as C

KEYS = ("index", "matrix", "frequency", "s11_co", "s11_cross", "s12", "s22")


# ─── CST 2024 COM redirection (no admin rights needed) ─────────────────────
def _force_cst_2024():
    import win32com.client
    orig = win32com.client.gencache.EnsureDispatch
    if getattr(orig, "_cst2024", False):
        return

    def dispatch(prog_id):
        return orig("CSTStudio.Application.2024" if prog_id == "CSTStudio.Application" else prog_id)
    dispatch._cst2024 = True
    win32com.client.gencache.EnsureDispatch = dispatch


# ─── Frequency grid ────────────────────────────────────────────────────────
def to_grid(freq, *curves):
    """Keep exactly the SUTD grid (FREQ_SAMPLES linear points). CST sometimes
    appends the frequencies at which its broadband sweep actually solved; those
    extra points are dropped (never interpolated)."""
    grid = np.linspace(C.FREQ_MIN, C.FREQ_MAX, C.FREQ_SAMPLES)
    freq = np.asarray(freq, float)
    idx = np.searchsorted(freq, grid)
    idx = np.clip(idx, 0, len(freq) - 1)
    near = np.where(np.abs(freq[np.maximum(idx - 1, 0)] - grid) < np.abs(freq[idx] - grid), np.maximum(idx - 1, 0), idx)
    if np.max(np.abs(freq[near] - grid)) > 1e-6:
        raise RuntimeError("CST result does not contain the full 1001-point frequency grid")
    return (grid,) + tuple(np.asarray(c)[near] for c in curves)


# ─── Sample files ──────────────────────────────────────────────────────────
def sample_path(i):
    return os.path.join(C.OUT_DIR, f"sample_{i:06d}.npz")


def sample_ok(path, design=None):
    """True if the file is a complete sample (all keys, 1001 finite points,
    and, if given, the same layout as the design file)."""
    try:
        with np.load(path) as d:
            if any(k not in d.files for k in KEYS):
                return False
            if len(d["frequency"]) != C.FREQ_SAMPLES:
                return False
            if not all(np.all(np.isfinite(d[k])) for k in ("s11_co", "s11_cross", "s12", "s22")):
                return False
            if design is not None and not np.array_equal(d["matrix"], design):
                return False
        return True
    except Exception:
        return False


def save_sample(i, matrix, freq, s11_co, s11_cross, s12, s22, seconds, worker):
    tmp = os.path.join(C.OUT_DIR, f".tmp_{i:06d}_{worker}.npz")
    np.savez_compressed(
        tmp, index=np.int64(i), matrix=matrix.astype(np.uint8),
        frequency=np.asarray(freq, float), s11_co=np.asarray(s11_co, np.complex128),
        s11_cross=np.asarray(s11_cross, np.complex128), s12=np.asarray(s12, np.complex128),
        s22=np.asarray(s22, np.complex128),
        substrate=C.SUB_NAME, substrate_eps=C.SUB_EPS, substrate_tand=C.SUB_TAND,
        substrate_thickness_mm=C.SUB_T_MM, conductor=C.CONDUCTOR, cell_mm=10.0,
        sim_seconds=float(seconds), worker=worker, finished=dt.datetime.now().isoformat(timespec="seconds"))
    if not sample_ok(tmp, matrix):
        os.remove(tmp)
        raise RuntimeError("written sample failed its check")
    os.replace(tmp, sample_path(i))            # atomic: the sample appears complete or not at all


# ─── Claims (lock files) ───────────────────────────────────────────────────
def claim(i, worker, stale_hours):
    lock = os.path.join(C.LOCK_DIR, f"{i:06d}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.time() - os.path.getmtime(lock) < stale_hours * 3600:
            return None                                  # another worker has it
        os.remove(lock)                                  # crashed worker: take over
        return claim(i, worker, stale_hours)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{worker} {socket.gethostname()} pid={os.getpid()} {dt.datetime.now().isoformat()}\n")
    return lock


def release(lock):
    try:
        os.remove(lock)
    except OSError:
        pass


# ─── Logs ──────────────────────────────────────────────────────────────────
def failures():
    path = os.path.join(C.LOG_DIR, "failed.csv")
    out = {}
    if os.path.exists(path):
        for row in csv.reader(open(path)):
            if row and row[0].isdigit():
                out[int(row[0])] = out.get(int(row[0]), 0) + 1
    return out


def log_failure(i, worker, err):
    path = os.path.join(C.LOG_DIR, "failed.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["index", "worker", "time", "error"])
        w.writerow([i, worker, dt.datetime.now().isoformat(timespec="seconds"), str(err).replace("\n", " ")[:500]])


def check_designs():
    d = np.load(C.DESIGNS_FILE)
    meta = json.load(open(C.DESIGNS_META))
    if hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest() != meta["sha256"]:
        sys.exit(f"{C.DESIGNS_FILE} does not match its checksum in {C.DESIGNS_META}; "
                 f"the samples would no longer refer to the right layouts.")
    stamp = os.path.join(C.OUT_DIR, "_dataset.json")     # ties the sample folder to this design file
    spec = dict(designs_sha256=meta["sha256"], substrate=C.SUB_NAME, eps=C.SUB_EPS, tand=C.SUB_TAND,
                thickness_mm=C.SUB_T_MM, conductor=C.CONDUCTOR, patch_mm=C.PATCH_SIZE_MM,
                padding_mm=C.PADDING_MM, patch_t_mm=C.PATCH_T_MM, ground_t_mm=C.GROUND_T_MM,
                freq_ghz=[C.FREQ_MIN, C.FREQ_MAX], points=C.FREQ_SAMPLES,
                convention="1 = copper", s_parameters="s11_co = r_xx (mode1->mode1), "
                "s11_cross = mode2->mode1, s12 = mode1<-mode2, s22 = mode2->mode2, Zmax Floquet port")
    if os.path.exists(stamp):
        if json.load(open(stamp)) != spec:
            sys.exit(f"{C.OUT_DIR} was made with different settings than fr4_config.py; "
                     f"use a new folder (change SUB_T_MM / TAG) instead of mixing datasets.")
    else:
        json.dump(spec, open(stamp, "w"), indent=2)
    return d


# ─── Main loop ─────────────────────────────────────────────────────────────
def main(args):
    for p in (C.OUT_DIR, C.LOCK_DIR, C.LOG_DIR, C.WORKSPACE_ROOT):
        os.makedirs(p, exist_ok=True)
    designs = check_designs()
    end = min(args.end, len(designs))
    todo = [i for i in range(args.start, end) if not os.path.exists(sample_path(i))]
    damaged = [i for i in range(args.start, end) if os.path.exists(sample_path(i))
               and not sample_ok(sample_path(i), designs[i])] if args.verify else []
    for i in damaged:
        print(f"  sample {i} is damaged; it will be simulated again")
        os.remove(sample_path(i))
    todo = sorted(set(todo) | set(damaged))
    tries = failures()
    todo = [i for i in todo if tries.get(i, 0) < args.max_tries]
    print(f"[{args.worker}] range {args.start}-{end}: {end - args.start - len(todo)} done, {len(todo)} to go "
          f"(substrate {C.SUB_NAME}, t = {C.SUB_T_MM:g} mm)", flush=True)
    if not todo:
        return

    _force_cst_2024()
    import cell_common
    cell_common.FREQ_MIN, cell_common.FREQ_MAX, cell_common.FREQ_SAMPLES = C.FREQ_MIN, C.FREQ_MAX, C.FREQ_SAMPLES
    ws = os.path.join(C.WORKSPACE_ROOT, f"{C.TAG}_{args.worker}")
    os.makedirs(ws, exist_ok=True)
    cell_common.build_base_template(ws, C.GRID_SIZE, C.PATCH_SIZE_MM, C.PADDING_MM, C.SUB_EPS, C.SUB_TAND,
                                    C.SUB_T_MM, ground_thickness=C.GROUND_T_MM, conductor=C.CONDUCTOR)

    done, t_run = 0, []
    log = open(os.path.join(C.LOG_DIR, f"progress_{args.worker}.log"), "a")
    try:
        for i in todo:
            if os.path.exists(sample_path(i)):
                continue
            lock = claim(i, args.worker, args.stale_hours)
            if lock is None:
                continue
            try:
                t0 = time.time()
                for attempt in range(4):              # CST may still hold the previous run's files
                    try:
                        freq, s11c, s11x, s12, s22 = cell_common.simulate_design(
                            ws, designs[i], C.GRID_SIZE, C.PATCH_SIZE_MM, C.PADDING_MM, C.SUB_T_MM,
                            patch_thickness=C.PATCH_T_MM, conductor=C.CONDUCTOR, return_all_modes=True)
                        break
                    except PermissionError:
                        if attempt == 3:
                            raise
                        time.sleep(30)
                sec = time.time() - t0
                freq, s11c, s11x, s12, s22 = to_grid(freq, s11c, s11x, s12, s22)
                save_sample(i, designs[i], freq, s11c, s11x, s12, s22, sec, args.worker)
                done += 1
                t_run.append(sec)
                left = len(todo) - done
                eta = np.mean(t_run[-50:]) * left / 3600
                msg = (f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  sample {i:6d} saved  {sec:5.0f} s  "
                       f"| this run {done}, left in range {left}, ETA {eta:.1f} h (this worker)")
                print(msg, flush=True)
                log.write(msg + "\n"); log.flush()
            except KeyboardInterrupt:
                raise
            except Exception as err:
                print(f"  sample {i} failed: {err}", flush=True)
                log_failure(i, args.worker, err)
            finally:
                release(lock)
    except KeyboardInterrupt:
        print(f"\n[{args.worker}] stopped by user after {done} new samples; run again to continue.")
    finally:
        log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=C.N_SIMULATE)
    ap.add_argument("--worker", default=socket.gethostname().lower(),
                    help="name of this worker (one CST workspace per worker)")
    ap.add_argument("--max_tries", type=int, default=3, help="skip designs that failed this many times")
    ap.add_argument("--stale_hours", type=float, default=2.0, help="take over locks older than this")
    ap.add_argument("--verify", action="store_true", help="re-check every finished sample on start")
    main(ap.parse_args())
