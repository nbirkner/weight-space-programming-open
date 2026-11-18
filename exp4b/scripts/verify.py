"""Independent re-check of saved edited networks on fresh seeds (not used by any run)."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, wsp, carenv
base = wsp.shown_weights()[0]
out = {}
for name, f in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    d = np.load(f); p = {k: d[k] for k in d.files}
    r1 = carenv.evaluate(p, 1024, seed=424242); r2 = carenv.evaluate(p, 1024, seed=99)
    g = carenv.evaluate(p, 1, seed=0, argmax=True)
    per = {k: int((np.abs(base[k] - p[k]) > 0).sum()) for k in base}
    out[name] = dict(file=f, fresh_seed_424242=r1, fresh_seed_99=r2, greedy_success=g["success"], changed=carenv.diff_count(base, p), per_tensor_changed={k: v for k, v in per.items() if v})
    print(name, r1["successes"], "/1024;", r2["successes"], "/1024; greedy", g["success"], "; changed", out[name]["changed"]["changed"], out[name]["per_tensor_changed"], flush=True)
json.dump(out, open(f"{wsp.ROOT}/results/verify.json", "w"), indent=1)
