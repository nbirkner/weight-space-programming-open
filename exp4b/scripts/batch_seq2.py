"""Continuation (Nov 17): attempts left after the $4.70 stop, same settings, hard stop $5.15. repeat_flash_01 counts as attempt 1. Attempts 2..10 use new tags/seeds.
Next run starts only if ledger total + 0.17 <= 4.70; chat() also refuses calls beyond 4.70 (WSP_LEDGER_HARD_STOP)."""
import subprocess, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import wsp
ROOT = wsp.ROOT; STOP = 5.15; env = dict(os.environ, WSP_LEDGER_HARD_STOP="5.15")
for i in range(2, 11):
    tag = f"repeat_seq_{i:02d}"
    if os.path.exists(f"{ROOT}/runs/{tag}/summary.json"): continue
    if wsp.spent() + 0.17 > STOP: print("budget stop before", tag, round(wsp.spent(), 4), flush=True); break
    cmd = [sys.executable, f"{ROOT}/scripts/run_tools.py", "--model", "deepseek-ai/DeepSeek-V4.1-Flash", "--tag", tag,
           "--max_turns", "30", "--passes", "30", "--max_tokens", "16384", "--feedback_rounds", "3", "--run_cap", "0.15",
           "--max_changed", "8600", "--wall", "600", "--seed", str(200 + i)]
    print("start", tag, "total", round(wsp.spent(), 4), flush=True)
    with open(f"{ROOT}/reports/{tag}.log", "w") as f:
        try: subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=5400, env=env)
        except subprocess.TimeoutExpired: print(tag, "timeout 90 min", flush=True)
    try: s = json.load(open(f"{ROOT}/runs/{tag}/summary.json"))
    except Exception as e: s = dict(status=f"no summary ({e})")
    print("done", tag, s.get("status"), (s.get("best_test") or {}).get("successes"), (s.get("best_changed") or {}).get("changed"), "run_cost", s.get("run_cost"), "total", round(wsp.spent(), 4), flush=True)
print("BATCH DONE total", round(wsp.spent(), 4), flush=True)
