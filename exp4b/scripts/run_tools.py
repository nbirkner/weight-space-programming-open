"""Exp 4b tool-assisted (+feedback, +compressed view) runner. Runs the LAST ```python block of each reply."""
import sys, os, re, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wsp, prompts
from sandbox import Sandbox

AUDIT_PAT = re.compile(r"\b(grad|value_and_grad|optax|torch|scipy\.optimize|minimize|autograd|backward|cma|nevergrad|random\.(normal|randn|uniform|rand)\b.*\b(W|param))", re.I)

def last_python_block(text):
    """The LAST code block in the reply: a ```python fence, or the model's native tool-call markup
    (DeepSeek emits <|DSML|invoke ...><parameter name="code">...</parameter> when it falls back to its own tool format)."""
    cands = [(m.start(), m.group(1)) for m in re.finditer(r"```(?:python|py)[^\n]*\n(.*?)```", text, re.S)]
    for m in re.finditer(r'<[｜|]DSML[｜|]\s*invoke name="[^"]*"[^>]*>(.*?)(?:</[｜|]DSML[｜|]\s*invoke>|$)', text, re.S):
        body = re.sub(r'</?[｜|]DSML[｜|]\s*parameter[^>]*>', '', m.group(1))
        body = re.sub(r'</?parameter[^>]*>', '', body)
        cands.append((m.start(), body))
    return max(cands)[1] if cands else None

def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--max_turns", type=int, default=30); ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--max_tokens", type=int, default=16384); ap.add_argument("--feedback_rounds", type=int, default=3)
    ap.add_argument("--run_cap", type=float, default=2.0); ap.add_argument("--wall", type=int, default=900)
    ap.add_argument("--mock", default=None); ap.add_argument("--max_changed", type=int, default=0); ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    rd = f"{wsp.ROOT}/runs/{a.tag}"; os.makedirs(rd, exist_ok=True)
    log = open(f"{rd}/transcript.jsonl", "a")
    def L(**k): k["ts"] = time.strftime("%H:%M:%S"); log.write(json.dumps(k) + "\n"); log.flush()
    mock = None
    if a.mock:
        import importlib.util; spec = importlib.util.spec_from_file_location("mockm", a.mock); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm); mock = mm.reply
    sb = Sandbox(f"{rd}/work", pass_budget=a.passes, max_changed=a.max_changed)
    sysmsg = "You are a careful engineer."
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": prompts.tools_prompt(a.passes, a.max_turns, a.max_changed)}]
    run_cost = 0.0; fb_done = 0; turns = 0; status = "running"; submissions = []
    base = wsp.shown_weights()[0]
    import carenv
    while turns < a.max_turns:
        turns += 1
        before = wsp.spent()
        try:
            r = wsp.chat(a.model, msgs, a.max_tokens, tag=f"{a.tag}/t{turns}", wall_s=a.wall, mock=mock, seed=(None if a.seed is None else a.seed * 1000 + turns))
        except RuntimeError as e:
            status = f"budget stop: {e}"; L(event="stop", why=str(e)); break
        run_cost += r.get("cost", 0.0)  # per-call cost (not ledger diff, which other parallel runs also move)
        content = strip_think(r["content"])
        L(event="reply", turn=turns, finish=r["finish"], error=r["error"], usage=r["usage"], wall=r["wall"], content=content, reasoning_chars=len(r["reasoning"]), ledger=wsp.spent())
        print(f"[{a.tag}] t{turns} finish={r['finish']} err={r['error']} ct={r['usage'].get('completion_tokens')} pt={r['usage'].get('prompt_tokens')} ledger=${wsp.spent():.3f} run=${run_cost:.3f}", flush=True)
        if r["error"] and not content:
            msgs.append({"role": "assistant", "content": "(no reply)"})
            msgs.append({"role": "user", "content": "Your last reply was cut off before any visible text (thinking limit). Keep the thinking shorter: a few sentences, then one python block."})
            if run_cost > a.run_cap: status = "run cap"; break
            continue
        if not content:
            msgs.append({"role": "assistant", "content": "(empty)"}); msgs.append({"role": "user", "content": "Your reply had no visible text (you probably ran out of thinking budget). Think less per reply: a few sentences, then one python block."})
            continue
        msgs.append({"role": "assistant", "content": content})
        code = last_python_block(content)
        if code is not None:
            flagged = bool(AUDIT_PAT.search(code))
            out = sb.run(code)
            o = out["out"]; o = o if len(o) <= 8000 else o[:4000] + "\n...[truncated]...\n" + o[-4000:]
            L(event="exec", turn=turns, code=code, out=o, secs=out.get("secs"), sim_calls=out.get("sim_calls"), passes_used=out.get("passes_used"), audit_flag=flagged)
            print(f"   exec {out.get('secs')}s sims={out.get('sim_calls')} used={out.get('passes_used')} flag={flagged} out={o[:200]!r}", flush=True)
            msgs.append({"role": "user", "content": f"Output (ran in {out.get('secs')} s; {a.passes - (out.get('passes_used') or 0)} simulation passes left; {a.max_turns - turns} replies left):\n```\n{o if o.strip() else '(nothing printed)'}\n```"})
        else:
            # final answer: optional edits block, else the saved edit
            blk, cut = wsp.last_edits_block(content)
            ep = f"{rd}/work/edit.npz"
            if blk is not None:
                p, rep = wsp.apply_edits(base, blk); src = "edits block"
            elif os.path.exists(ep):
                d = np.load(ep); p = {k: d[k] for k in d.files}; rep = None; src = "save_edit"
            else:
                p = None
            if p is not None and a.max_changed and carenv.diff_count(base, p)["changed"] > a.max_changed:
                L(event="final_refused", turn=turns, changed=carenv.diff_count(base, p)["changed"]); p = None
            if p is None:
                res = None; L(event="final", turn=turns, note="no edit saved")
            else:
                res = carenv.evaluate(p, 256, seed=0, trace=True)
                sub = dict(turn=turns, src=src, search=res["success"], search_successes=res["successes"], changed=carenv.diff_count(base, p)["changed"], rep=rep)
                np.savez(f"{rd}/submission_t{turns}.npz", **p); submissions.append(sub)
                L(event="final", **sub, trace=res["trace_ep0"])
                print(f"   SUBMISSION t{turns}: search {res['successes']}/256 changed={sub['changed']}", flush=True)
            if res is not None and res["success"] >= 0.999: status = "solved on search"; break
            if fb_done >= a.feedback_rounds: status = "feedback rounds used"; break
            fb_done += 1
            if res is None:
                msgs.append({"role": "user", "content": f"No edit was saved, so nothing was scored. Please continue: use save_edit(params). ({a.max_turns - turns} replies left)"})
            else:
                msgs.append({"role": "user", "content": prompts.FEEDBACK.format(
                    res=f"success {res['successes']}/256 ({100*res['success']:.1f} %), red {100*res['red']:.1f} %, timeout {100*res['timeout']:.1f} %, mean return {res['return']:.3f}.",
                    trace=res["trace_ep0"], extra="", passes=a.passes - json.load(open(f"{rd}/work/sim_calls.json"))["calls"] if os.path.exists(f"{rd}/work/sim_calls.json") else a.passes, turns=a.max_turns - turns)})
        if run_cost > a.run_cap: status = f"run cap ${a.run_cap}"; L(event="stop", why=status); break
    else:
        status = "max turns"
    # if the run ended without a final reply, score the last saved edit
    ep = f"{rd}/work/edit.npz"
    if not submissions and os.path.exists(ep):
        d = np.load(ep); p = {k: d[k] for k in d.files}; res = carenv.evaluate(p, 256, seed=0)
        np.savez(f"{rd}/submission_last_saved.npz", **p)
        submissions.append(dict(turn=turns, src="save_edit (run ended)", search=res["success"], search_successes=res["successes"], changed=carenv.diff_count(base, p)["changed"]))
    sb.close()
    best = max(submissions, key=lambda s: s["search"]) if submissions else None
    summary = dict(tag=a.tag, model=a.model, status=status, turns=turns, run_cost=round(run_cost, 4), submissions=submissions, best=best, args=vars(a))
    if best is not None:
        f = f"{rd}/submission_t{best['turn']}.npz" if os.path.exists(f"{rd}/submission_t{best['turn']}.npz") else f"{rd}/submission_last_saved.npz"
        d = np.load(f); p = {k: d[k] for k in d.files}
        summary["best_file"] = f
        summary["best_test"] = carenv.evaluate(p, 1024, seed=12345)
        summary["best_argmax"] = carenv.evaluate(p, 1, seed=0, argmax=True)["success"]
        summary["best_changed"] = carenv.diff_count(base, p)
    json.dump(summary, open(f"{rd}/summary.json", "w"), indent=1, default=str)
    print("SUMMARY", json.dumps({k: summary.get(k) for k in ["tag", "status", "turns", "run_cost", "best", "best_test", "best_changed"]}, default=str), flush=True)

if __name__ == "__main__":
    main()
