# Keeps Windows awake while long training jobs run (the equivalent of `caffeinate`): asks the OS not to sleep for as long as
# this process lives. It changes no power setting; stop it (or close the window) and the normal sleep rules apply again.
#   python tools/keep_awake.py            # until you stop it
#   python tools/keep_awake.py --while-training   # exits when no train_dqn.py / train_ppo.py process is running
import argparse
import ctypes
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def training_running():
    import psutil
    # a training job, or the supervisor that restarts jobs (so a short gap between a stop and a restart does not end the keep-awake)
    keys = ("train_dqn.py", "train_ppo.py", "ext_supervisor.py")
    return any(any(k in " ".join(p.info["cmdline"] or []) for k in keys) for p in psutil.process_iter(["cmdline"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--while-training", action="store_true")
    a = ap.parse_args()
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        while True:
            time.sleep(60)
            if a.while_training and not training_running():
                break
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
