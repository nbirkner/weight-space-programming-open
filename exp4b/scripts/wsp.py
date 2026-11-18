"""Shared pieces for Exp 4b: shown (rounded) weights, the paper's edit language, ledger, Together client."""
import os, re, json, time, threading
import numpy as np
from paths import ROOT, PROMPT
SHAPES = {"embed.W": (679, 128), "embed.b": (128,), "actor_head.W": (128, 16), "actor_head.b": (16,)}
for k in range(5): SHAPES[f"actor{k}.W"] = (128, 128); SHAPES[f"actor{k}.b"] = (128,)

def shown_weights(prompt_path=PROMPT):
    """Parse the weights exactly as shown to Astra (integers x scale)."""
    t = open(prompt_path).read()
    i0 = t.index("THE WEIGHTS (complete actor)")
    body = t[i0:]
    out, scales = {}, {}
    hdr = re.compile(r"^(\S+) \(([\d x]+)\), scale (1e-\d+):", re.M)
    hs = list(hdr.finditer(body))
    for j, h in enumerate(hs):
        name = h.group(1); sc = float(h.group(3)); scales[name] = sc
        seg = body[h.end(): hs[j + 1].start() if j + 1 < len(hs) else len(body)]
        lines = [l for l in seg.strip().split("\n") if l.strip()]
        shp = SHAPES[name]
        if len(shp) == 2:
            rows = []
            for l in lines[1:] if "one line per row" in lines[0] else lines:
                if ":" not in l: continue
                rows.append([int(v) for v in l.split(":", 1)[1].split()])
            a = np.array(rows, dtype=np.float64)
        else:
            a = np.array([int(v) for l in lines for v in l.split()], dtype=np.float64)
        assert a.shape == shp, (name, a.shape, shp)
        out[name] = a * sc
    return out, scales, t[:i0]

# ---------------- edit language ----------------
STMT = re.compile(r"^\s*([a-z_0-9]+\.(?:W|b))\s*(\[[^\]]*\])?\s*(=|\+=|-=|\*=)\s*(.+)$")
def last_edits_block(text):
    blocks = re.findall(r"```edits[^\n]*\n(.*?)```", text, re.S)
    if not blocks:
        m = re.search(r"```edits[^\n]*\n(.*)$", text, re.S)  # unterminated = cut off
        return (m.group(1), True) if m else (None, False)
    return blocks[-1], False

def _idx(s, shape):
    if s is None: return tuple(slice(None) for _ in shape)
    parts = [p.strip() for p in s.strip()[1:-1].split(",")]
    if len(parts) != len(shape): raise ValueError("index rank")
    ix = []
    for p in parts:
        if ":" in p:
            a, b = (p.split(":") + [""])[:2]
            ix.append(slice(int(a) if a.strip() else None, int(b) if b.strip() else None))
        else: ix.append(int(p))
    return tuple(ix)

def apply_edits(params, block):
    p = {k: v.copy() for k, v in params.items()}
    rep = dict(statements=0, applied=0, entries_written=0, errors=[])
    if block is None: return p, rep
    # join continuation lines (lists spanning lines)
    stmts, cur = [], ""
    for raw in block.split("\n"):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip(): continue
        cur = (cur + " " + line.strip()) if cur else line.strip()
        if cur.count("[") <= cur.count("]"): stmts.append(cur); cur = ""
    if cur: stmts.append(cur)
    for s in stmts:
        rep["statements"] += 1
        try:
            if s.startswith("ZERO "):
                n = s.split()[1]; p[n][...] = 0; rep["applied"] += 1; rep["entries_written"] += p[n].size; continue
            m = STMT.match(s)
            if not m: raise ValueError("unparsed")
            n, ix, op, val = m.groups()
            if n not in p: raise ValueError("unknown tensor " + n)
            sl = _idx(ix, p[n].shape)
            val = val.strip()
            v = np.array(json.loads(val), dtype=np.float64) if val.startswith("[") else float(val)
            if np.any(np.abs(v) > 1e6) or not np.all(np.isfinite(v)): raise ValueError("value out of range")
            tgt = p[n][sl]
            if op == "=": p[n][sl] = v
            elif op == "+=": p[n][sl] = tgt + v
            elif op == "-=": p[n][sl] = tgt - v
            elif op == "*=": p[n][sl] = tgt * v
            rep["applied"] += 1; rep["entries_written"] += int(np.size(tgt))
        except Exception as e:
            rep["errors"].append(f"{s[:80]} -> {e}")
    return p, rep

# ---------------- ledger ----------------
PRICES = {}  # $/1M tokens, filled from Together /v1/models (results/models.json)
for m in json.load(open(f"{ROOT}/results/models.json")):
    if m.get("pricing"): PRICES[m["id"]] = m["pricing"]
HARD_STOP = float(os.environ.get("WSP_LEDGER_HARD_STOP", "9.50"))
_lock = threading.Lock()
LEDGER = f"{ROOT}/ledger.json"
def ledger():
    if os.path.exists(LEDGER): return json.load(open(LEDGER))
    return {"budget": 10.0, "hard_stop": HARD_STOP, "total": 0.0, "calls": []}
def cost_of(model, usage):
    pr = PRICES[model]; pt = usage.get("prompt_tokens", 0); ct = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or usage.get("cached_tokens") or 0
    return ((pt - cached) * pr["input"] + cached * pr.get("cached_input", pr["input"]) + ct * pr["output"]) / 1e6
def record(model, usage, tag, estimated=False):
    with _lock:
        L = ledger(); c = cost_of(model, usage)
        L["calls"].append(dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), model=model, tag=tag, usage=usage, cost=round(c, 6), estimated=estimated))
        L["total"] = round(L["total"] + c, 6); json.dump(L, open(LEDGER, "w"), indent=1); return L["total"]
def spent(): return ledger()["total"]
def worst_case(model, prompt_tokens, max_tokens):
    pr = PRICES[model]; return (prompt_tokens * pr["input"] + max_tokens * pr["output"]) / 1e6

# ---------------- Together client (streaming, wall cap) ----------------
def chat(model, messages, max_tokens, tag, wall_s=900, temperature=0.7, est_prompt_tokens=None, mock=None, seed=None):
    if mock is not None:
        return mock(messages)
    est = est_prompt_tokens or int(sum(len(m["content"]) for m in messages) / 2.6)
    if spent() + worst_case(model, est, max_tokens) > HARD_STOP:
        raise RuntimeError(f"budget stop: spent {spent():.3f} + worst case {worst_case(model, est, max_tokens):.3f} > {HARD_STOP}")
    import requests
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
                stream_options={"include_usage": True})
    if seed is not None: body["seed"] = int(seed)
    hdr = {"Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"], "Content-Type": "application/json"}
    content, reasoning, usage, finish, err = [], [], None, None, None
    t0 = time.time()
    try:
        with requests.post("https://api.together.xyz/v1/chat/completions", json=body, headers=hdr, stream=True, timeout=(20, 240)) as r:
            if r.status_code != 200:
                err = f"HTTP {r.status_code}: {r.text[:300]}"
            else:
                r.encoding = "utf-8"
                for raw in r.iter_lines(decode_unicode=True):
                    if time.time() - t0 > wall_s: err = f"wall cap {wall_s}s"; break
                    if not raw or not raw.startswith("data:"): continue
                    data = raw[5:].strip()
                    if data == "[DONE]": break
                    try: ch = json.loads(data)
                    except Exception: continue
                    if ch.get("usage"): usage = ch["usage"]
                    for c in ch.get("choices") or []:
                        d = c.get("delta") or {}
                        if d.get("content"): content.append(d["content"])
                        rs = d.get("reasoning") or d.get("reasoning_content")
                        if rs: reasoning.append(rs)
                        if c.get("finish_reason"): finish = c["finish_reason"]
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    content, reasoning = "".join(content), "".join(reasoning)
    estimated = usage is None
    if usage is None:  # charge a conservative estimate for cut-off streams
        usage = dict(prompt_tokens=est, completion_tokens=int((len(content) + len(reasoning)) / 3.0) + 50)
    call_cost = 0.0
    if not (err and err.startswith("HTTP") and not content and not reasoning):
        total = record(model, usage, tag, estimated); call_cost = cost_of(model, usage)
    else:
        total = spent()
    return dict(content=content, reasoning=reasoning, usage=usage, finish=finish, error=err, wall=round(time.time() - t0, 1),
                estimated=estimated, ledger_total=total, cost=call_cost)
