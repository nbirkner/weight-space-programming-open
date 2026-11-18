"""Kinetix l/h2_car_ramp evaluator for the exp4 actor (numpy-dict params).
Rebuilt Nov 16 2026 for Exp 4b. Matches the paper's prompt: symbolic flat obs (679),
multi-discrete actions (16 logits, 6 entries), frame_skip 2, 256 steps, sampled actions."""
import os, json
import paths
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax, jax.numpy as jnp
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import load_from_json_file
from kinetix.environment.utils import static_env_params_from_size

LEVEL = "l/h2_car_ramp"
_env_state, _sp, _ep = load_from_json_file(LEVEL)
SP = static_env_params_from_size("l")
EP = _ep.replace(dense_reward_scale=0.2)  # the paper prompt states 0.2 x (d_prev - d)
ENV = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None, EP, SP, auto_reset=False)
NAMES = ["embed", "actor0", "actor1", "actor2", "actor3", "actor4", "actor_head"]
SEGS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 14), (14, 16)]

def load_base(path=None):
    if path is None:
        path = paths.require_upstream('exp4_fix_car', 'networks', 'ppo_final_seed0.npz')
    d = np.load(path); p = "params/"; g = p + "GeneralActorCriticRNN_0/"
    m = {"embed": p + "Dense_0", "actor_head": g + "Dense_0"}
    for k in range(5): m[f"actor{k}"] = g + f"DenseAndActivation_{k}/Dense_0"
    out = {}
    for n, key in m.items():
        out[n + ".W"] = d[key + "/kernel"].astype(np.float64)
        out[n + ".b"] = d[key + "/bias"].astype(np.float64)
    return out

def load_npz(path):
    d = np.load(path); return {k: d[k].astype(np.float64) for k in d.files}

def forward(params, obs):
    x = jnp.clip(jnp.nan_to_num(obs, nan=0., posinf=0., neginf=0.), -10, 10)
    h = jax.nn.relu(x @ params["embed.W"] + params["embed.b"])
    for k in range(5):
        h = jnp.tanh(h @ params[f"actor{k}.W"] + params[f"actor{k}.b"])
    return h @ params["actor_head.W"] + params["actor_head.b"]

def _sample(rng, logits, argmax):
    acts = []
    for i, (a, b) in enumerate(SEGS):
        l = logits[a:b]
        r = jax.random.fold_in(rng, i)
        acts.append(jnp.where(argmax, jnp.argmax(l), jax.random.categorical(r, l)))
    return jnp.stack(acts)

def _episode(params, rng, argmax):
    obs, state = ENV.reset(rng, EP, override_reset_state=_env_state)
    def body(c, t):
        obs, state, done, outcome, ret, rng = c
        rng, r1, r2 = jax.random.split(rng, 3)
        logits = forward(params, obs)
        act = _sample(r1, logits, argmax)
        obs2, state2, rew, d2, info = ENV.step(r2, state, act, EP)
        alive = ~done
        # outcome: 0 running, 1 success, 2 red, 3 timeout/invalid
        term = (rew >= 0.5) * 1 + (rew <= -0.5) * 2
        new_out = jnp.where(alive & d2, jnp.where(term > 0, term, 3), outcome)
        ret = ret + jnp.where(alive, rew, 0.)
        rec = (state2.polygon.position[5], state2.circle.position[0], state2.circle.position[1], act[0], jax.nn.softmax(logits[0:3]))
        obs = jnp.where(alive, obs2, obs); state = jax.tree.map(lambda a, b: jnp.where(alive, a, b), state2, state)
        return (obs, state, done | d2, new_out, ret, rng), rec
    init = (obs, state, jnp.array(False), jnp.array(0), jnp.array(0.), rng)
    (obs, state, done, outcome, ret, _), rec = jax.lax.scan(body, init, jnp.arange(EP.max_timesteps))
    outcome = jnp.where(outcome == 0, 3, outcome)
    return outcome, ret, rec

_run = jax.jit(jax.vmap(_episode, in_axes=(None, 0, None)), static_argnums=(2,))

def evaluate(params, n=256, seed=0, argmax=False, trace=False):
    p = {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}
    rngs = jax.random.split(jax.random.PRNGKey(seed), n)
    out, ret, rec = _run(p, rngs, argmax)
    out = np.asarray(out); ret = np.asarray(ret)
    res = {"episodes": int(n), "successes": int((out == 1).sum()), "success": float((out == 1).mean()),
           "red": float((out == 2).mean()), "timeout": float((out == 3).mean()), "return": float(ret.mean())}
    if trace:
        body, g, w, a0, pr = [np.asarray(x)[0] for x in rec]
        lines = ["step | body x,y | green wheel x,y | wheel x,y | action0 | p(0/1/2)"]
        for t in list(range(0, 256, 16)) + [255]:
            lines.append(f"{t:4d} | {body[t,0]:.2f},{body[t,1]:.2f} | {g[t,0]:.2f},{g[t,1]:.2f} | {w[t,0]:.2f},{w[t,1]:.2f} | {int(a0[t])} | {pr[t,0]:.2f}/{pr[t,1]:.2f}/{pr[t,2]:.2f}")
        res["trace_ep0"] = "\n".join(lines)
    return res

def evaluate_full(params, search_n=256, test_n=1024):
    return {"search": evaluate(params, search_n, seed=0), "test": evaluate(params, test_n, seed=12345),
            "argmax": evaluate(params, 1, seed=0, argmax=True)}

def diff_count(base, new):
    ch = sum(int((np.abs(base[k] - new[k]) > 0).sum()) for k in base); tot = sum(base[k].size for k in base)
    return {"changed": ch, "entries": tot, "frac": ch / tot}
