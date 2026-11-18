import json, statistics as st, sys, os
from collections import defaultdict
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
L = [json.loads(l) for l in open(os.path.join(ROOT, "results/paid/ledger.jsonl"))]
final = {}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_exp1c import score
RES = os.path.join(ROOT, "results/paid")
def fixenc(t):
    try: return t.encode("latin-1").decode("utf-8")
    except Exception: return t
for r in L:  # official cell = last attempt; rescore from saved content with the current strict scorer
    if r["key"] not in final or r["attempt"] >= final[r["key"]]["attempt"]:
        d = json.load(open(f"{RES}/{r['key']}__a{r['attempt']}.json"))
        r = dict(r); r["verdict_live"] = r["verdict"]
        r.update(score(fixenc(d["content"]), r["truth"]))
        final[r["key"]] = r
groups = defaultdict(list)
for r in final.values():
    groups[(r["model"], r["cond"], r["cap"])].append(r)
costs = defaultdict(float)
for r in L: costs[(r["model"], r["cond"], r["cap"])] += r["cost"]
order = [("flash","heatmap",c) for c in (2048,4096,8192,16384)] + [("flash",c,24576) for c in ("w1_only","ascii","glyphs","topk8","heatmap_anon","heatmap_shuffled")] + [("qwen",c,24576) for c in ("heatmap_anon","heatmap_shuffled")]
def med(x): return st.median(x) if x else None
print("| Model | Condition | cap | right/attempts | correct/scored | correct | miss | abstain | incomplete | median input | median reasoning | median wall | cost |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in order:
    g = groups.get(k, [])
    if not g: continue
    v = defaultdict(int)
    for r in g: v[r["verdict"]] += 1
    if k[1] == "heatmap_shuffled":
        cell = f"declined {v['declined']}, named<50% {v['named_low']}, named≥50% {v['named']}, incomplete {v['incomplete']}"
        print(f"| {k[0]} | {k[1]} | {k[2]} | {cell} | | | | | | {med([r['prompt_tokens'] for r in g])} | {med([r['reasoning_tokens'] for r in g if r['reasoning_tokens']])} | {med([r['wall'] for r in g])}s | ${costs[k]:.4f} |")
        continue
    sc = v["correct"] + v["miss"] + v["abstain"]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {v['correct']}/{len(g)} | {v['correct']}/{sc} | {v['correct']} | {v['miss']} | {v['abstain']} | {v['incomplete']} | {med([r['prompt_tokens'] for r in g])} | {med([r['reasoning_tokens'] for r in g if r['reasoning_tokens']])} | {med([r['wall'] for r in g])}s | ${costs[k]:.4f} |")
print()
print("total spent $%.4f over %d calls (%d estimated-usage)" % (sum(r["cost"] for r in L), len(L), sum(r["usage_estimated"] for r in L)))
if "-v" in sys.argv:
    for k in order:
        for r in sorted(groups.get(k, []), key=lambda r: r["key"]):
            print(k[1], k[2], r["game"], r["file"][-14:], "s%d" % r["seed"], r["verdict"], repr((r["game_line"] or "")[:70]), r["conf"], r["finish"], r["error"])
