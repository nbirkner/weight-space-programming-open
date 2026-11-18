"""Leakage check: winning assisted setup on the hidden-unit-permuted network, neutral prompt example. 3 sequential attempts, seeds 301-303."""
import subprocess, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import wsp
ROOT = wsp.ROOT; STOP = 5.80; env = dict(os.environ, WSP_LEDGER_HARD_STOP="5.80")
start = wsp.spent()
for i in range(1, 4):
    tag = f"leak_perm_{i:02d}"
    if os.path.exists(f"{ROOT}/runs/{tag}/summary.json"): continue
    if wsp.spent() + 0.17 > STOP or wsp.spent() - start + 0.17 > 0.55: print("budget stop before", tag, round(wsp.spent(), 4), flush=True); break
    cmd = [sys.executable, f"{ROOT}/scripts/run_tools_perm.py", "--model", "deepseek-ai/DeepSeek-V4.1-Flash", "--tag", tag,
           "--max_turns", "30", "--passes", "30", "--max_tokens", "16384", "--feedback_rounds", "3", "--run_cap", "0.15",
           "--max_changed", "8600", "--wall", "600", "--seed", str(300 + i)]
    print("start", tag, "total", round(wsp.spent(), 4), flush=True)
    with open(f"{ROOT}/reports/{tag}.log", "w") as f:
        try: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=5400, env=env)
        except subprocess.TimeoutExpired: print(tag, "timeout 90 min", flush=True)
    try: s = json.load(open(f"{ROOT}/runs/{tag}/summary.json"))
    except Exception as e: s = dict(status=f"no summary ({e})")
    print("done", tag, s.get("status"), (s.get("best_test") or {}).get("successes"), (s.get("best_changed") or {}).get("changed"), "run_cost", s.get("run_cost"), "total", round(wsp.spent(), 4), flush=True)
print("LEAK DONE total", round(wsp.spent(), 4), flush=True)
