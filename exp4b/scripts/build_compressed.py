"""Approach 3: a compressed, no-tools version of the paper's prompt (~1/10 the tokens).
Keeps the paper's task text, scene, rules, network formula, result and edit format; replaces the 171,664-number dump with:
embed.W rows for the 23 observation entries that change in this scene, each embed unit's constant offset, per-unit summaries
of actor0..actor4, and the full actor_head."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, wsp, prompts
W, sc, head = wsp.shown_weights()
names = np.load(f"{wsp.ROOT}/results/obs_names.npy")
var = [27,29,52,53,54,55,59,64,65,234,235,236,237,241,246,247,252,253,254,255,259,264,265]
x0 = None
import carenv, jax
obs, _ = carenv.ENV.reset(jax.random.PRNGKey(0), carenv.EP, override_reset_state=carenv._env_state); x0 = np.asarray(obs)
const = [i for i in range(679) if i not in var]
c = x0[const] @ W["embed.W"][const] + W["embed.b"]
i0 = head.index("EDIT FORMAT"); task_and_scene = head[:i0]
task_and_scene = task_and_scene.replace("Your edits are applied to the network shown below", "Your edits are applied to the full network (summarised below)")
L = [task_and_scene, head[i0:].strip(), "",
     "THE WEIGHTS, COMPRESSED. We do not show all 171,664 numbers. In this fixed scene only 23 of the 679 observation entries ever change; every other entry is a constant, so its contribution to each embed unit is a constant. Edits may still touch any entry of any tensor (indices refer to the full tensors).", "",
     "Observation entries that change (index | meaning | value at start):"]
for i in var: L.append(f"{i} | {names[i]} | {x0[i]:.3f}")
L += ["", "embed.W rows for those 23 entries (128 numbers each, scale 1e-2):"]
for i in var: L.append(f"{i}: " + " ".join(str(int(round(v / 0.01))) for v in W["embed.W"][i]))
L += ["", "c (128): each embed unit's constant pre-activation, i.e. sum over all constant entries of x_i * embed.W[i, j] plus embed.b[j] (scale 1e-3). embed_j = relu(c_j + sum over the 23 changing entries of x_i * embed.W[i, j]):",
      " ".join(str(int(round(v / 0.001))) for v in c)]
for k in range(5):
    Wk, bk = W[f"actor{k}.W"], W[f"actor{k}.b"]
    L += ["", f"actor{k} (128 x 128 W, 128 b). Per output unit j: b[j] (1e-3) | sum|W[:, j]| (1e-2) | largest-magnitude inputs (unit:weight x1e-3):"]
    for j in range(128):
        top = np.argsort(-np.abs(Wk[:, j]))[:4]
        L.append(f"{j}: {int(round(bk[j]/0.001))} | {int(round(np.abs(Wk[:, j]).sum()/0.01))} | " + " ".join(f"{t}:{int(round(Wk[t, j]/0.001))}" for t in top))
L += ["", "actor_head.W (128 x 16), scale 1e-3: one line per row (row i)"]
for i in range(128): L.append(f"{i}: " + " ".join(str(int(round(v / 0.001))) for v in W["actor_head.W"][i]))
L += ["", "actor_head.b (16), scale 1e-3:", " ".join(str(int(round(v / 0.001))) for v in W["actor_head.b"])]
t = "\n".join(L)
open(f"{wsp.ROOT}/results/compressed_prompt.txt", "w").write(t)
print(len(t), "chars")
