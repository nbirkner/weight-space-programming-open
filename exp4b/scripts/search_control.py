"""Non-LLM search controls for Exp 4b (no API calls, $0).
Same network as shown to the models (rounded weights), same evaluator (carenv.evaluate), same 8,600-weight cap,
same strict test (1,024 episodes, seed 12345, success = reaching the goal).
  --method random : random sparse perturbations + hill climbing on the training score.
  --method greedy : the same kind of edit the LLM wrote (take over 2 hidden units as dedicated threshold
                    channels through all layers, rewire the entry-0 motor logits to them), parameters chosen by
                    greedy coordinate search on the training score.
Training score = (successes, mean return) on 64 sampled episodes, seed 0, compared lexicographically."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wsp, carenv
ROOT = wsp.ROOT; CAP = 8600
BASE = wsp.shown_weights()[0]
NAMES = list(BASE.keys())
VARY = [27, 29, 52, 53, 54, 55, 59, 64, 65, 234, 235, 236, 237, 241, 246, 247, 252, 253, 254, 255, 259, 264, 265]
# value ranges from results/obs_summary.txt (min, max seen)
RANGE = {}
for line in open(f"{ROOT}/results/obs_summary.txt"):
    p = [x.strip() for x in line.split("|")]
    if len(p) == 5 and p[0].isdigit(): RANGE[int(p[0])] = (float(p[3]), float(p[4]))

class Scorer:
    def __init__(self, budget, n=64):
        self.budget, self.n, self.calls, self.hist = budget, n, 0, []
        self.best = None; self.best_p = None; self.first_full = None
    def __call__(self, p):
        if self.calls >= self.budget: raise StopIteration
        self.calls += 1
        r = carenv.evaluate(p, self.n, seed=0)
        s = (r["successes"], round(r["return"], 6)); self.hist.append(s)
        if self.best is None or s > self.best:
            self.best, self.best_p = s, {k: v.copy() for k, v in p.items()}
        if self.first_full is None and r["successes"] == self.n:
            self.first_full = dict(call=self.calls, params={k: v.copy() for k, v in p.items()})
        return s

def changed(p): return carenv.diff_count(BASE, p)["changed"]

def random_search(sc, rng):
    cur = {k: v.copy() for k, v in BASE.items()}; cs = sc(cur)
    std = {k: max(float(BASE[k].std()), 0.05) for k in NAMES}
    while True:
        cand = {k: v.copy() for k, v in cur.items()}
        m = int(rng.integers(1, 65))
        for _ in range(m):
            k = NAMES[int(rng.integers(len(NAMES)))]; a = cand[k]
            idx = tuple(int(rng.integers(s)) for s in a.shape)
            a[idx] += rng.normal() * std[k] * 10 ** rng.uniform(-0.5, 1.5)
        if changed(cand) > CAP: continue
        s = sc(cand)
        if s >= cs: cur, cs = cand, s

def build(units, prm):
    """Template edit: unit u in every hidden layer becomes a dedicated channel = relu(K*sign*(x_i - thr)) passed through
    tanh layers with gain 10; motor logits (entry 0) read those channels."""
    p = {k: v.copy() for k, v in BASE.items()}; K = 100.0
    for u, q in zip(units, prm["units"]):
        i, sg, thr = VARY[q["inp"]], q["sign"], q["thr"]
        p["embed.W"][:, u] = 0.0; p["embed.W"][i, u] = sg * K; p["embed.b"][u] = -sg * K * thr
        for k in range(5):
            W = p[f"actor{k}.W"]; W[:, u] = 0.0; W[u, u] = 10.0; p[f"actor{k}.b"][u] = 0.0
    H = p["actor_head.W"]
    if prm["zero_others"]: H[:, 0:3] = 0.0
    for u, q in zip(units, prm["units"]): H[u, 0:3] = q["head"]
    p["actor_head.b"][0:3] = prm["hb"]
    return p

HEADV = [-10.0, 0.0, 10.0]; HBV = [-10.0, -5.0, 0.0, 5.0, 10.0]
def thr_grid(q):
    lo, hi = RANGE[VARY[q["inp"]]]; return list(np.linspace(lo, hi, 7))

def greedy_search(sc, rng):
    units = sorted(int(u) for u in rng.choice(128, 2, replace=False))
    def rand_unit():
        q = dict(inp=int(rng.integers(len(VARY))), sign=float(rng.choice([-1.0, 1.0])))
        q["thr"] = float(rng.choice(thr_grid(q))); q["head"] = [float(rng.choice(HEADV)) for _ in range(3)]; return q
    prm = dict(units=[rand_unit(), rand_unit()], zero_others=bool(rng.integers(2)), hb=[float(rng.choice(HBV)) for _ in range(3)])
    import copy
    cs = sc(build(units, prm)); log = [dict(units=units)]
    def try_set(setter, values):
        nonlocal prm, cs
        best_v, best_s = None, cs
        for v in values:
            c = copy.deepcopy(prm); setter(c, v)
            s = sc(build(units, c))
            if s > best_s: best_v, best_s = v, s
        if best_v is not None: setter(prm, best_v); cs = best_s
    while True:
        for j in range(2):
            try_set(lambda c, v, j=j: c["units"][j].__setitem__("inp", v), range(len(VARY)))
            try_set(lambda c, v, j=j: c["units"][j].__setitem__("sign", v), [-1.0, 1.0])
            try_set(lambda c, v, j=j: c["units"][j].__setitem__("thr", v), thr_grid(prm["units"][j]))
            for a in range(3):
                try_set(lambda c, v, j=j, a=a: c["units"][j]["head"].__setitem__(a, v), HEADV)
        for a in range(3):
            try_set(lambda c, v, a=a: c["hb"].__setitem__(a, v), HBV)
        try_set(lambda c, v: c.__setitem__("zero_others", v), [False, True])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=120); a = ap.parse_args()
    tag = f"{a.method}_s{a.seed}_b{a.budget}"; out = f"{ROOT}/control/{tag}"; os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed); sc = Scorer(a.budget); t0 = time.time()
    try: (random_search if a.method == "random" else greedy_search)(sc, rng)
    except StopIteration: pass
    res = dict(tag=tag, method=a.method, seed=a.seed, budget=a.budget, train_episodes_per_call=sc.n, calls=sc.calls,
               secs=round(time.time() - t0), best_train=sc.best, best_changed=changed(sc.best_p),
               first_full_train_call=(sc.first_full or {}).get("call"), hist=sc.hist)
    np.savez(f"{out}/best.npz", **sc.best_p)
    res["best_test"] = carenv.evaluate(sc.best_p, 1024, seed=12345)
    if sc.first_full is not None:
        res["first_full_test"] = carenv.evaluate(sc.first_full["params"], 1024, seed=12345)
        res["first_full_changed"] = changed(sc.first_full["params"])
    json.dump(res, open(f"{out}/result.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "hist"}, default=str), flush=True)

if __name__ == "__main__":
    main()
