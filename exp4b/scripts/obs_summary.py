"""Compressed view (approach 3): which of the 679 observation entries change in this fixed scene, and what they are."""
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, carenv, wsp
W, _, _ = wsp.shown_weights()
def names():
    n = []
    pf = ["pos.x","pos.y","vel.x","vel.y","inv_mass","inv_inertia","density","tanh(angvel/10)","role.none","role.green","role.blue","role.red","sin(angle)","cos(angle)","friction","restitution","is_circle_flag0"] + [f"vert{i//2}.{'xy'[i%2]}" for i in range(8)] + ["<=3verts"]
    for s in [0,4,5,6,7,8,9,10,11]: n += [f"poly{s}.{x}" for x in pf]
    cf = pf[:16] + ["radius","is_circle"]
    for s in [12,13,14,15]: n += [f"circle{s}.{x}" for x in cf]
    jf = ["active","fixed","b_pos.x","b_pos.y","a_pos.x","a_pos.y","sin_ref","cos_ref","motor_speed","motor_power","motor_on","has_limits"]+[f"bind{i}" for i in range(4)]+["sin_min","cos_min","sin_max","cos_max","min-ref","max-ref"]+[f"b_onehot{i}" for i in range(16)]+[f"a_onehot{i}" for i in range(16)]
    for s in range(6): n += [f"joint{s}.{x}" for x in jf]
    tf = ["active","pos.x","pos.y","bind0","bind1","sin","cos","power"]+[f"shape{i}" for i in range(16)]
    for s in range(2): n += [f"thruster{s}.{x}" for x in tf]
    n += ["gravity_y/10"]; assert len(n) == 679, len(n); return n
N = names()
pj = {k: jnp.asarray(v, jnp.float32) for k, v in W.items()}
O = []
for pol in ["base", 0, 1, 2]:
    for seed in range(3):
        obs, st = carenv.ENV.reset(jax.random.PRNGKey(0), carenv.EP, override_reset_state=carenv._env_state)
        rng = np.random.default_rng(seed)
        for t in range(256):
            O.append(np.asarray(obs))
            if pol == "base": a0 = int(rng.choice(3, p=np.asarray(jax.nn.softmax(carenv.forward(pj, obs)[0:3]))))
            else: a0 = pol if rng.random() < 0.8 else int(rng.integers(3))
            obs, st, r, d, info = carenv.ENV.step(jax.random.PRNGKey(t), st, jnp.array([a0,0,0,0,0,0]), carenv.EP)
            if bool(d): break
O = np.array(O); x0 = O[0]
var = np.where((O.max(0) - O.min(0)) > 1e-3)[0]
const_nz = np.where(((O.max(0) - O.min(0)) <= 1e-3) & (np.abs(x0) > 0))[0]
lines = [f"Observation entries that change during episodes in this scene ({len(var)} of 679; all others are constant):",
         "index | meaning | value at start | min | max seen (base policy and fixed-action rollouts)"]
for i in var: lines.append(f"{i} | {N[i]} | {x0[i]:.3f} | {O[:,i].min():.3f} | {O[:,i].max():.3f}")
lines.append(f"\n{len(const_nz)} entries are constant and non-zero (shape properties, joint settings, gravity); the rest are 0.")
# effective constant bias: contribution of constant entries to embed pre-activation
cb = x0[const_nz] @ W["embed.W"][const_nz] + W["embed.b"]
lines.append("Because those entries never change, embed unit j effectively has a constant offset c_j = sum_i x_i*embed.W[i,j] + embed.b[j] over the constant entries; the varying entries move it from there.")
open(f"{wsp.ROOT}/results/obs_summary.txt", "w").write("\n".join(lines))
np.save(f"{wsp.ROOT}/results/obs_names.npy", np.array(N))
print("\n".join(lines))
