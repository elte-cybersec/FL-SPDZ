#!/usr/bin/env python3
import subprocess, time, sys, os, signal


NUM_PARTIES  = 2          # total parties in the MPC run
PARTY_ID     = 1          # this launcher is party 1
EXECUTABLE   = "./semi2k-party.x" # or mascot-party.x, etc.
RESTART_DELAY = 2         # seconds between restarts; set 0 to disable auto‑restart


if len(sys.argv) < 2:
        print("Usage: python server_rl_mtd.py <algorithm>")
        sys.exit(1)
algorithm = sys.argv[1]

PROGRAM_MAPPING = {
    "ppo": "fedavg_ppo",
    "a2c": "fedavg_a2c",
    "envelope": "fedavg_envelope",
    "eupg": "fedavg_eupg"
}

PROGRAM = PROGRAM_MAPPING.get(algorithm.lower(), "fedavg_eupg")   # byte‑code name (no .mpc/.bc)

def launch_party(total: int, pid: int, prog: str):
    """Launch a single MP-SPDZ party and return the Popen handle."""
    cmd = [
        EXECUTABLE,
        "-p", str(pid),
        "-N", str(total),
        # "-pn", str(5100),
        prog,          # ExternalIO on stdin/stdout; remove if not needed
    ]
    print("[launcher] starting:", " ".join(cmd))
    # Use a new process group so we can kill children on Ctrl‑C
    return subprocess.Popen(cmd, preexec_fn=os.setsid)

try:
    while True:
        proc = launch_party(NUM_PARTIES, PARTY_ID, PROGRAM)
        rc = proc.wait()               # wait until the party exits
        print(f"[launcher] party exited with code {rc}")

        if RESTART_DELAY <= 0:
            break                      # no auto‑restart, quit

        print(f"[launcher] restarting in {RESTART_DELAY}s ...")
        time.sleep(RESTART_DELAY)

except KeyboardInterrupt:
    print("\n[launcher] Ctrl‑C received, terminating party")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    sys.exit(0)
