"""Cost section: actual Together spend from ledger.json, and the same token counts repriced at published
gpt-6-astra and Claude Fable 5.2 prices (checked Nov 17 2026). Writes results/costs.json."""
import json, statistics, glob, os
from paths import ROOT
L = json.load(open(f"{ROOT}/ledger.json"))
def toks(u):
    pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
    ca = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("cached_tokens") or 0
    rs = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return pt, ca, ct, rs
# $/1M. Cache-write premiums are ignored (we do not know which tokens would be writes), so cached pricing is a lower bound.
def astra(pt, ca, ct):  # >272K input: 2x input and cache, 1.5x output, for the whole request
    lg = pt > 272_000; m_in, m_out = (2.0, 1.5) if lg else (1.0, 1.0)
    return dict(cached=((pt - ca) * 10 * m_in + ca * 1.0 * m_in + ct * 50 * m_out) / 1e6, uncached=(pt * 10 * m_in + ct * 50 * m_out) / 1e6)
def fable(pt, ca, ct):  # no long-context surcharge
    return dict(cached=((pt - ca) * 8 + ca * 0.20 + ct * 40) / 1e6, uncached=(pt * 8 + ct * 40) / 1e6)
runs = {}
for c in L["calls"]:
    t = c["tag"].split("/")[0]; r = runs.setdefault(t, dict(calls=0, cost=0.0, pt=0, cached=0, ct=0, reasoning=0, astra=0.0, astra_nc=0.0, fable=0.0, fable_nc=0.0, estimated_calls=0, model=set()))
    pt, ca, ct, rs = toks(c["usage"]); a = astra(pt, ca, ct); f = fable(pt, ca, ct)
    r["calls"] += 1; r["cost"] += c["cost"]; r["pt"] += pt; r["cached"] += ca; r["ct"] += ct; r["reasoning"] += rs
    r["astra"] += a["cached"]; r["astra_nc"] += a["uncached"]; r["fable"] += f["cached"]; r["fable_nc"] += f["uncached"]
    r["estimated_calls"] += bool(c.get("estimated")); r["model"].add(c["model"])
for t, r in runs.items():
    r["model"] = sorted(r["model"])
    s = f"{ROOT}/runs/{t}/summary.json"
    if os.path.exists(s):
        S = json.load(open(s)); r["status"] = S.get("status"); r["test"] = (S.get("best_test") or {}).get("successes")
json.dump(runs, open(f"{ROOT}/results/costs.json", "w"), indent=1)
# Exp 1c ledger (for repricing; tokens only)
e1c = dict(cost=0, pt=0, ct=0, reasoning=0, astra=0, fable=0, calls=0)
p = os.path.join(os.path.dirname(ROOT), "exp1c", "results", "paid", "ledger.jsonl")
if os.path.exists(p):
    for line in open(p):
        d = json.loads(line); pt = d.get("prompt_tokens", 0) or 0; ct = d.get("completion_tokens", 0) or 0
        e1c["calls"] += 1; e1c["cost"] += d.get("cost", 0) or 0; e1c["pt"] += pt; e1c["ct"] += ct; e1c["reasoning"] += d.get("reasoning_tokens", 0) or 0
        e1c["astra"] += astra(pt, 0, ct)["uncached"]; e1c["fable"] += fable(pt, 0, ct)["uncached"]
json.dump(e1c, open(f"{ROOT}/results/costs_exp1c.json", "w"), indent=1)
print(json.dumps({t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for t, r in runs.items()}, indent=0)[:6000])
print("exp1c", {k: round(v, 4) for k, v in e1c.items()})
