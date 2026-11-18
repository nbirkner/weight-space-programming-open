"""Exp 1c runner (rebuilt 2026-11-16 from Notion notes; the Exp 1b runner was not in this repository).

Usage: python run_exp1c.py --mock     # free dry run, no network
       python run_exp1c.py --paid     # real Together calls, hard $5 cap
Key comes from env TOGETHER_API_KEY and is never printed or written.
"""
import argparse, json, os, re, sys, threading, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = {
    "flash": dict(id="deepseek-ai/DeepSeek-V4.1-Flash", pin=0.30, pout=1.20, tpc_in=0.558, workers=3, worst_out_mult=1.0),
    "qwen": dict(id="Qwen/Qwen3.7-Plus", pin=0.32, pout=1.28, tpc_in=0.753, workers=4, worst_out_mult=2.45),  # v2: 4 workers; worst = 60k out tokens  # 1b saw 42,644 > 24,576 on Qwen
}
HARD_CAP = 4.50  # resume rule 2026-11-16 12:13 PT: total incl. earlier spend must stay under $4.50; never $5
WALL_S = {"flash": 900, "qwen": 1500}  # v2: Qwen streams ~25-40 tok/s and ignores max_tokens
READ_TIMEOUT = 300  # v2: was 180s; Qwen stalls >180s between chunks
SYSTEM = "Answer the user's request."

# ---------------- strict scorer ----------------
GAMES = {
    "breakout": [r"break\s*-?\s*out", r"arkanoid"],
    "pong": [r"\bpong\b"],
    "pacman": [r"pac\s*-?\s*man"],
    "freeway": [r"freeway", r"frogger"],
}
ABSTAIN = [r"\bunknown\b", r"cannot (?:be )?determin", r"can't (?:be )?determin", r"\bunclear\b", r"undetermin",
           r"not identifiable", r"no (?:specific|recogni[sz]able|identifiable) game", r"^\W*none\b", r"cannot identify",
           r"insufficient", r"indeterminate", r"random(?:ly)? (?:initiali[sz]ed|weights|network)", r"untrained", r"no game"]

def game_line(content):
    lines = (content or "").splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*[#>*_\-\s]*Game\s*[*_]*\s*:\s*[*_]*\s*(.*)$", line, re.I)
        if m:
            rest = m.group(1).strip().strip("*_ ").strip()
            if rest:
                return rest
            # answer on the following line ("Game:\nBreakout")
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    if re.match(r"^\s*[#>*_\-\s]*(Description|Confidence|Reasoning|Observations)\s*[*_]*\s*:", nxt, re.I):
                        return ""
                    return nxt.strip().strip("*_ ").strip()
            return ""
    return None

def confidence(content):
    m = re.search(r"Confidence\s*[*_]*\s*:\s*[*_]*\s*~?(\d{1,3})", content or "", re.I)
    return int(m.group(1)) if m else None

def primary_game(line):
    best = None
    for g, pats in GAMES.items():
        for p in pats:
            m = re.search(p, line, re.I)
            if m and (best is None or m.start() < best[1]):
                best = (g, m.start())
    return best[0] if best else None

def score(content, truth):
    """Strict: only the visible final content's first Game: line counts."""
    gl = game_line(content)
    if gl is None:
        return dict(verdict="incomplete", game_line=None, primary=None, conf=None)
    prim = primary_game(gl)
    conf = confidence(content)
    abst = prim is None and any(re.search(p, gl, re.I) for p in ABSTAIN)
    if truth == "decoy":
        if abst:
            v = "declined"
        elif conf is not None and conf < 50:
            v = "named_low"
        else:
            v = "named"
        return dict(verdict=v, game_line=gl, primary=prim, conf=conf)
    if prim == truth:
        v = "correct"
    elif abst:
        v = "abstain"
    else:
        v = "miss"
    return dict(verdict=v, game_line=gl, primary=prim, conf=conf)

# ---------------- cells ----------------
def cells():
    man = json.load(open(f"{ROOT}/prompts/manifest.json"))
    by = {}
    for r in man:
        r = dict(r); by.setdefault(r.pop("cond"), []).append(r)
    out = []
    order = ["breakout", "pong", "pacman", "freeway"]
    srt = lambda L: sorted(L, key=lambda r: (order.index(r["game"]), r["file"]))
    for cap in (2048, 4096, 8192, 16384):  # A. budget curve
        for r in srt(by["heatmap"]):
            for s in (1, 2, 3):
                out.append(dict(block="A", model="flash", cond="heatmap", cap=cap, seed=s, **r))
    for cond in ("w1_only", "ascii", "glyphs", "topk8"):  # B. smaller reps
        for r in srt(by[cond]):
            for s in (1, 2, 3):
                out.append(dict(block="B", model="flash", cond=cond, cap=24576, seed=s, **r))
    for model, blk in (("flash", "C"), ("qwen", "Q")):  # C / Qwen add-on
        for r in srt(by["heatmap_anon"]):
            for s in (1, 2, 3):
                out.append(dict(block=blk, model=model, cond="heatmap_anon", cap=24576, seed=s, **r))
        for r in srt(by["heatmap_shuffled"]):
            out.append(dict(block=blk, model=model, cond="heatmap_shuffled", cap=24576, seed=1, **r))
    for c in out:
        c["key"] = f"{c['model']}__{c['cond']}__cap{c['cap']}__{os.path.basename(c['file'])[:-4]}__s{c['seed']}"
    return out

# ---------------- ledger / budget ----------------
class Ledger:
    def __init__(self, path):
        self.path = path; self.lock = threading.Lock(); self.spent = 0.0; self.reserved = 0.0; self.stopped = False
        if os.path.exists(path):
            for line in open(path):
                self.spent += json.loads(line)["cost"]
    def reserve(self, worst):
        with self.lock:
            if self.stopped or self.spent + self.reserved + worst > HARD_CAP:
                self.stopped = True
                return False
            self.reserved += worst
            return True
    def settle(self, worst, rec):
        with self.lock:
            self.reserved -= worst; self.spent += rec["cost"]
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return self.spent

# ---------------- API call ----------------
OUT_TPC = {"flash": 0.28, "qwen": 0.28}  # updated empirically from completed calls

def call_together(model, prompt, cap, seed):
    import requests
    m = MODELS[model]
    body = dict(model=m["id"], messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=cap, temperature=0.7, seed=seed, stream=True, stream_options={"include_usage": True})
    hdr = {"Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"], "Content-Type": "application/json"}
    content, reasoning, usage, finish, err = [], [], None, None, None
    t0 = time.time()
    try:
        with requests.post("https://api.together.xyz/v1/chat/completions", json=body, headers=hdr, stream=True,
                           timeout=(15, READ_TIMEOUT)) as r:
            if r.status_code != 200:
                return dict(content="", reasoning="", usage=None, finish=None, error=f"HTTP {r.status_code}: {r.text[:300]}", wall=time.time() - t0)
            r.encoding = "utf-8"
            for raw in r.iter_lines(decode_unicode=True):
                if time.time() - t0 > WALL_S[model]:
                    err = f"wall cap {WALL_S[model]}s"; break
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)
                except Exception:
                    continue
                if ch.get("usage"):
                    usage = ch["usage"]
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content"): content.append(d["content"])
                    rs = d.get("reasoning") or d.get("reasoning_content")
                    if rs: reasoning.append(rs)
                    if c.get("finish_reason"): finish = c["finish_reason"]
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    if err is None and finish is None:
        err = "stream ended without finish_reason"
    return dict(content="".join(content), reasoning="".join(reasoning), usage=usage, finish=finish, error=err, wall=time.time() - t0)

def call_mock(model, prompt, cap, seed, truth):
    time.sleep(0.01)
    rnd = random.Random(hash((model, prompt[-200:], cap, seed)))
    reason = "thinking " * rnd.randint(500, 3000)
    k = rnd.random()
    if k < 0.15:
        return dict(content="", reasoning=reason, usage=dict(prompt_tokens=int(len(prompt) * 0.56), completion_tokens=cap), finish="length", error=None, wall=1.0)
    if k < 0.2:
        return dict(content="", reasoning=reason, usage=None, finish=None, error="ConnectionError: Read timed out (mock)", wall=1.0)
    g = {"breakout": "Breakout", "pong": "Pong (wall variant)", "pacman": "Pac-Man", "freeway": "Frogger"}.get(truth, "Unknown; the weights look unstructured")
    if k > 0.85: g = "A grid variant of Pong, Breakout, or Snake"
    content = f"Observations: ...\nReasoning: ...\n**Game:** {g}\nDescription: ...\nConfidence: {rnd.randint(30, 99)}%\n"
    ct = rnd.randint(1000, min(cap, 12000))
    return dict(content=content, reasoning=reason, usage=dict(prompt_tokens=int(len(prompt) * 0.56), completion_tokens=ct), finish="stop", error=None, wall=1.0)

def cost_of(model, pt, ct):
    m = MODELS[model]
    return (pt * m["pin"] + ct * m["pout"]) / 1e6

def run_cell(c, ledger, mock, resdir, attempt):
    m = MODELS[c["model"]]
    prompt = open(c["file"]).read()
    in_est = len(prompt) * m["tpc_in"]
    worst = cost_of(c["model"], in_est * 1.05, c["cap"] * m["worst_out_mult"])
    if not ledger.reserve(worst):
        return None
    r = call_mock(c["model"], prompt, c["cap"], c["seed"], c["truth"]) if mock else call_together(c["model"], prompt, c["cap"], c["seed"])
    u = r["usage"]
    est = False
    if u and u.get("prompt_tokens") is not None:
        pt, ct = u["prompt_tokens"], u.get("completion_tokens", 0)
    else:
        est = True
        pt = int(in_est)
        ct = int((len(r["content"]) + len(r["reasoning"])) * OUT_TPC[c["model"]])
    cost = cost_of(c["model"], pt, ct)
    sc = score(r["content"], c["truth"])
    if r["error"] and sc["verdict"] != "incomplete":
        pass  # a visible Game: line was already streamed; strict scorer keeps it
    rec = dict(key=c["key"], attempt=attempt, block=c["block"], model=c["model"], model_id=m["id"], cond=c["cond"], cap=c["cap"],
               game=c["game"], net=c["net"], truth=c["truth"], seed=c["seed"], file=os.path.basename(c["file"]),
               prompt_tokens=pt, completion_tokens=ct, reasoning_tokens=((u or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
               usage_estimated=est, cost=round(cost, 6), finish=r["finish"], error=r["error"], wall=round(r["wall"], 1),
               content_chars=len(r["content"]), reasoning_chars=len(r["reasoning"]), **sc, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    spent = ledger.settle(worst, rec)
    if not est and (len(r["content"]) + len(r["reasoning"])) > 2000 and ct:
        OUT_TPC[c["model"]] = 0.7 * OUT_TPC[c["model"]] + 0.3 * ct / (len(r["content"]) + len(r["reasoning"]))
    os.makedirs(resdir, exist_ok=True)
    json.dump(dict(rec=rec, content=r["content"], reasoning=r["reasoning"], usage=u), open(f"{resdir}/{c['key']}__a{attempt}.json", "w"))
    print(f"[{rec['ts']}] {c['key']} a{attempt} -> {sc['verdict']} ({sc['game_line']!r:.60}) pt={pt} ct={ct} ${cost:.4f} finish={r['finish']} err={r['error']} wall={rec['wall']}s | spent ${spent:.4f}", flush=True)
    return rec

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true"); g.add_argument("--paid", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma list of keys")
    a = ap.parse_args()
    tag = "mock" if a.mock else "paid"
    if a.paid and not os.environ.get("TOGETHER_API_KEY"):
        sys.exit("TOGETHER_API_KEY not set")
    resdir = f"{ROOT}/results/{tag}"; os.makedirs(resdir, exist_ok=True)
    ledger = Ledger(f"{resdir}/ledger.jsonl")
    done = set(); tries = {}
    def transport_fail(rr):
        return rr["verdict"] == "incomplete" and (rr["error"] or rr["finish"] is None)
    if os.path.exists(ledger.path):
        for line in open(ledger.path):
            rr = json.loads(line)
            if rr.get("interrupted"):
                continue  # killed by us mid-stream; spend counted, cell not done
            tries[rr["key"]] = max(tries.get(rr["key"], 0), rr["attempt"])
            if rr["attempt"] >= 2 or not transport_fail(rr):
                done.add(rr["key"])
    todo = [c for c in cells() if c["key"] not in done]
    if a.only: todo = [c for c in todo if c["key"] in a.only.split(",")]
    if a.limit: todo = todo[: a.limit]
    print(f"{len(todo)} cells to run; ledger so far ${ledger.spent:.4f}", flush=True)

    def worker(c):
        attempts = tries.get(c["key"], 0) + 1  # 1 = fresh cell, 2 = the single retry
        rec = run_cell(c, ledger, a.mock, resdir, attempts)
        # one retry max, only for transport failures (no visible answer + error/drop). Length caps are results.
        if rec and attempts == 1 and rec["error"] and rec["verdict"] == "incomplete" and rec["finish"] != "length":
            rec = run_cell(c, ledger, a.mock, resdir, 2)
        return rec

    pools = {m: ThreadPoolExecutor(MODELS[m]["workers"]) for m in MODELS}
    futs = [pools[c["model"]].submit(worker, c) for c in todo]
    for f in as_completed(futs):
        f.result()
    for p in pools.values(): p.shutdown()
    print(f"DONE. ledger total ${ledger.spent:.4f}; stopped_by_cap={ledger.stopped}", flush=True)

if __name__ == "__main__":
    main()
