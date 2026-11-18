"""Sequential repeat batch of the capped repair setup (same as repair_flash_1). Batch cap $1.20; next run starts only if
enough budget remains for a full run, and each run's own cap shrinks to fit what is left."""
import os
import subprocess, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import wsp
ROOT = wsp.ROOT; CAP = 1.20; start = wsp.spent(); res = []
for i in range(1, 11):
    used = wsp.spent() - start; left = CAP - used
    if left < 0.06 or wsp.spent() > 9.0: print("stop: batch budget", used, flush=True); break
    run_cap = min(0.10, left - 0.04)
    tag = f"repeat_flash_{i:02d}"
    cmd = [sys.executable, f"{ROOT}/scripts/run_tools.py", "--model", "deepseek-ai/DeepSeek-V4.1-Flash", "--tag", tag,
           "--max_turns", "30", "--passes", "30", "--max_tokens", "16384", "--feedback_rounds", "3", "--run_cap", f"{run_cap:.3f}",
           "--max_changed", "8600", "--wall", "600", "--seed", str(100 + i)]
    print("start", tag, "run_cap", round(run_cap, 3), "batch used", round(used, 4), flush=True)
    with open(f"{ROOT}/reports/{tag}.log", "w") as f:
        try: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=5400)
        except subprocess.TimeoutExpired: print(tag, "run timeout 90 min", flush=True)
    try: s = json.load(open(f"{ROOT}/runs/{tag}/summary.json"))
    except Exception as e: s = dict(tag=tag, status=f"no summary ({e})")
    res.append(s); print("done", tag, s.get("status"), (s.get("best_test") or {}).get("successes"), (s.get("best_changed") or {}).get("changed"), "batch used", round(wsp.spent() - start, 4), flush=True)
    json.dump(dict(batch_start_total=start, batch_used=wsp.spent() - start, runs=res), open(f"{ROOT}/results/repeat_batch.json", "w"), indent=1, default=str)
print("BATCH DONE used", round(wsp.spent() - start, 4), "total", round(wsp.spent(), 4), flush=True)
