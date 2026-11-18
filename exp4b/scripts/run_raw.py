"""Exp 4b raw-weights runs: the paper's exact prompt. rounds=1 is the strict 1:1 (approach 4 when max_tokens is raised);
rounds>1 adds feedback rounds (approach 2): after each reply we score the edit and send the result back; later edits
apply on top of the current edited network."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, wsp, carenv, paths
from run_tools import strip_think
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--max_tokens", type=int, default=32768)
ap.add_argument("--wall", type=int, default=1500); ap.add_argument("--prompt_file", default=None); ap.add_argument("--run_cap", type=float, default=1.5)
a = ap.parse_args()
rd = f"{wsp.ROOT}/runs/{a.tag}"; os.makedirs(rd, exist_ok=True)
prompt = open(a.prompt_file or wsp.PROMPT).read()
EST = 356000 if not a.prompt_file else int(len(prompt) / 2.4)
instr = open(paths.require_upstream('exp4_fix_car', 'runs', 'oneshot_repair', 'instructions.md')).read().strip()
msgs = [{"role": "system", "content": instr}, {"role": "user", "content": prompt}]
base = wsp.shown_weights()[0]; cur = {k: v.copy() for k, v in base.items()}
subs = []; run_cost = 0.0; log = open(f"{rd}/transcript.jsonl", "a")
for rnd in range(1, a.rounds + 1):
    before = wsp.spent()
    try:
        r = wsp.chat(a.model, msgs, a.max_tokens, tag=f"{a.tag}/r{rnd}", wall_s=a.wall, est_prompt_tokens=EST + 12000 * (rnd - 1))
    except RuntimeError as e:
        print("STOP", e); break
    run_cost += wsp.spent() - before
    c = strip_think(r["content"])
    blk, cut = wsp.last_edits_block(c)
    p, rep = wsp.apply_edits(cur, blk)
    res = carenv.evaluate(p, 256, seed=0, trace=True) if blk else dict(successes=0, success=0.0, red=0, timeout=1, **{"return": 0}, trace_ep0="(no edits block: network unchanged)")
    ch = carenv.diff_count(base, p)["changed"]
    log.write(json.dumps(dict(round=rnd, finish=r["finish"], error=r["error"], usage=r["usage"], wall=r["wall"], content=c, reasoning=r["reasoning"][-20000:], edit_report=rep, search=res["success"], changed=ch)) + "\n"); log.flush()
    print(f"[{a.tag}] r{rnd} finish={r['finish']} err={r['error']} usage={r['usage']} block={blk is not None} cut={cut} rep={ {k: rep[k] for k in ['statements','applied']} } errs={len(rep['errors'])} search={res['successes']}/256 changed={ch} ledger=${wsp.spent():.3f}", flush=True)
    if blk:
        cur = p; np.savez(f"{rd}/edit_r{rnd}.npz", **p); subs.append(dict(round=rnd, search=res["success"], search_successes=res["successes"], changed=ch))
    if res["success"] >= 0.999 or rnd == a.rounds or run_cost > a.run_cap: break
    msgs.append({"role": "assistant", "content": c if c else "(no visible reply)"})
    if blk:
        fb = (f"Your edits were applied and the edited network was scored on 256 episodes (actions sampled): success {res['successes']}/256 ({100*res['success']:.1f} %), "
              f"red {100*float(res['red']):.1f} %, timeout {100*float(res['timeout']):.1f} %, mean return {float(res['return']):.3f}.\nOne sampled episode, every 16 steps:\n{res['trace_ep0']}\n")
    else:
        fb = "Your reply had no edits block (it may have been cut off by the length limit), so the network is unchanged.\n"
    fb += "You get another try (no tools). Any new edits block is applied ON TOP of the network as it is now (your earlier edits stay). Keep the reasoning brief and make sure the reply ends with an edits block."
    msgs.append({"role": "user", "content": fb})
summary = dict(tag=a.tag, model=a.model, rounds_run=rnd, run_cost=round(run_cost, 4), submissions=subs, args=vars(a))
if subs:
    best = max(subs, key=lambda s: s["search"]); d = np.load(f"{rd}/edit_r{best['round']}.npz"); p = {k: d[k] for k in d.files}
    summary.update(best=best, best_test=carenv.evaluate(p, 1024, seed=12345), best_changed=carenv.diff_count(base, p))
json.dump(summary, open(f"{rd}/summary.json", "w"), indent=1, default=str)
print("SUMMARY", json.dumps({k: summary.get(k) for k in ["tag", "rounds_run", "run_cost", "best", "best_test"]}, default=str), flush=True)
