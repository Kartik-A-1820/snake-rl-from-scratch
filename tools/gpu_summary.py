# Summarises results/bench/gpu_temp_log.csv (written by tools/ext_supervisor.py): temperatures, clocks, and how often the driver
# reported a thermal or power throttle. Bit meanings (nvidia-smi clocks_throttle_reasons): 0x04 SW power cap, 0x08 HW slowdown,
# 0x20 SW thermal slowdown, 0x40 HW thermal slowdown, 0x80 HW power brake.
import csv
import statistics
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "results" / "bench" / "gpu_temp_log.csv"
rows = list(csv.DictReader(p.open()))
if not rows:
    raise SystemExit("no samples")
temp = [int(r["temp_c"]) for r in rows]
clk = [int(r["sm_mhz"]) for r in rows]
thr = [int(r["throttle_hex"], 16) for r in rows]
n = len(rows)
longest = cur = 0
for t in temp:
    cur = cur + 1 if t >= 87 else 0
    longest = max(longest, cur)
print(f"{n} samples from {rows[0]['time']} to {rows[-1]['time']} ({n * 30 / 3600:.1f} h at 30 s)")
print(f"GPU temperature: mean {statistics.mean(temp):.1f} C, median {statistics.median(temp)} C, max {max(temp)} C; samples >= 87 C: {sum(t >= 87 for t in temp)} (longest streak {longest})")
print(f"SM clock: mean {statistics.mean(clk):.0f} MHz, min {min(clk)}, max {max(clk)}")
for bit, name in ((0x04, "SW power cap"), (0x20, "SW thermal slowdown"), (0x40, "HW thermal slowdown"), (0x08, "HW slowdown"), (0x80, "HW power brake")):
    k = sum(1 for t in thr if t & bit)
    hot = [int(r["temp_c"]) for r, t in zip(rows, thr) if t & bit]
    print(f"{name:22s}: {k:5d} samples ({100 * k / n:.0f}%)" + (f", GPU temp during it: {min(hot)}-{max(hot)} C" if hot else ""))
