"""Function-preserving random permutation of the hidden units in all six hidden layers (embed, actor0..actor4).
New unit j of layer L is old unit perm[L][j]. Saves networks/perm_shown.npz (+ perm.json)."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, wsp
ROOT = wsp.ROOT
base = wsp.shown_weights()[0]
rng = np.random.default_rng(31337)
layers = ["embed"] + [f"actor{k}" for k in range(5)]
P = {L: rng.permutation(128) for L in layers}
q = {k: v.copy() for k, v in base.items()}
prev = None
for L in layers:
    W = base[f"{L}.W"]
    if prev is not None: W = W[P[prev], :]
    q[f"{L}.W"] = W[:, P[L]].copy(); q[f"{L}.b"] = base[f"{L}.b"][P[L]].copy(); prev = L
q["actor_head.W"] = base["actor_head.W"][P[prev], :].copy()
np.savez(f"{ROOT}/networks/perm_shown.npz", **q)
img = {L: {str(o): int(np.where(P[L] == o)[0][0]) for o in (0, 1, 2, 122, 123, 124, 125, 126, 127)} for L in layers}
json.dump({"seed": 31337, "perm_new_to_old": {L: P[L].tolist() for L in layers}, "images_of_old_units": img}, open(f"{ROOT}/networks/perm.json", "w"), indent=1)
print(json.dumps(img))
