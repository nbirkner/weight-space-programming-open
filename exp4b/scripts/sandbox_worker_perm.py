"""Persistent python worker run inside bubblewrap (no network, no env vars).
Reads JSON {"code":...} lines on stdin, execs in one persistent namespace, replies JSON {"out":...}."""
import sys, json, io, contextlib, traceback, os
sys.path.insert(0, "/exp4b/scripts")
import numpy as np
_real_stdout = sys.stdout
import carenv, wsp
_d = np.load("/exp4b/shown.npz"); _W = {k: _d[k].astype(np.float64) for k in _d.files}  # permuted network (leakage check)
_SIM = {"calls": 0, "turn_calls": 0}
MAX_SIM_PER_RUN = 4
PASS_BUDGET = int(os.environ.get("PASS_BUDGET", "30"))
MAX_CHANGED = int(os.environ.get("MAX_CHANGED", "0"))

def W_copy():
    return {k: v.copy() for k, v in _W.items()}

def apply_edits(edits_text, params=None):
    """Apply edit statements (the paper's edit language) to a copy of params (default: the shown network)."""
    p, rep = wsp.apply_edits(params if params is not None else _W, edits_text)
    return p, rep

def simulate(params, n=64, trace=True):
    """Run n episodes (n <= 256) with actions sampled from the policy. Returns a dict with success rate etc."""
    if _SIM["turn_calls"] >= MAX_SIM_PER_RUN:
        raise RuntimeError(f"simulate() limit: at most {MAX_SIM_PER_RUN} calls per code run")
    if _SIM["calls"] >= PASS_BUDGET:
        raise RuntimeError(f"simulation budget used up ({PASS_BUDGET} passes)")
    n = int(min(max(n, 1), 256)); _SIM["turn_calls"] += 1; _SIM["calls"] += 1
    return carenv.evaluate(params, n, seed=7000 + _SIM["calls"], trace=trace)

def simulate_greedy(params):
    """One episode with the most-likely action each step (deterministic). Counts toward the simulate() limit."""
    if _SIM["turn_calls"] >= MAX_SIM_PER_RUN:
        raise RuntimeError(f"simulate() limit: at most {MAX_SIM_PER_RUN} calls per code run")
    if _SIM["calls"] >= PASS_BUDGET:
        raise RuntimeError(f"simulation budget used up ({PASS_BUDGET} passes)")
    _SIM["turn_calls"] += 1; _SIM["calls"] += 1
    return carenv.evaluate(params, 1, seed=0, argmax=True, trace=True)

def observe(params=None, actions=None, n_steps=64):
    """Roll the scene forward with a fixed list of action-entry-0 values (0/1/2) or with the policy's greedy actions,
    returning the observation vectors (n_steps x 679) and per-step entry-0 logits. With a fixed action list it is free;
    with actions=None (the network drives) it costs one pass."""
    import jax, jax.numpy as jnp
    if actions is None:
        if _SIM["calls"] >= PASS_BUDGET: raise RuntimeError(f"simulation budget used up ({PASS_BUDGET} passes)")
        _SIM["calls"] += 1; _SIM["turn_calls"] += 1
    p = params if params is not None else _W
    pj = {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}
    obs, state = carenv.ENV.reset(jax.random.PRNGKey(0), carenv.EP, override_reset_state=carenv._env_state)
    O, L = [], []
    for t in range(n_steps):
        lg = np.asarray(carenv.forward(pj, obs)); O.append(np.asarray(obs)); L.append(lg[0:3])
        a0 = int(np.argmax(lg[0:3])) if actions is None else int(actions[t] if t < len(actions) else actions[-1])
        act = jnp.array([a0, 0, 0, 0, 0, 0])
        obs, state, rew, done, info = carenv.ENV.step(jax.random.PRNGKey(t), state, act, carenv.EP)
        if bool(done): break
    return np.array(O), np.array(L)

def forward(params, obs):
    """Logits (16) for one observation vector (679) or a batch (N x 679)."""
    import jax.numpy as jnp
    return np.asarray(carenv.forward({k: jnp.asarray(v, jnp.float32) for k, v in params.items()}, jnp.asarray(obs, jnp.float32)))

def save_edit(params):
    """Save params as your current edited network (this is what gets scored when you finish)."""
    for k, v in _W.items():
        a = np.asarray(params[k], dtype=np.float64)
        if a.shape != v.shape or not np.all(np.isfinite(a)) or np.abs(a).max() > 1e6: raise ValueError("bad tensor " + k)
    ch = sum(int((np.abs(np.asarray(params[k]) - _W[k]) > 0).sum()) for k in _W)
    if MAX_CHANGED and ch > MAX_CHANGED:
        raise ValueError(f"edit refused: {ch} weights differ from the shown network; the limit for this run is {MAX_CHANGED}")
    np.savez("/work/edit.npz", **{k: np.asarray(params[k], dtype=np.float64) for k in _W})
    print(f"[saved edit: {ch} of 171664 weights differ from the shown network]")

def passes_left():
    return PASS_BUDGET - _SIM["calls"]

NS = dict(save_edit=save_edit, passes_left=passes_left, np=np, W=_W, W_copy=W_copy, apply_edits=apply_edits, simulate=simulate, simulate_greedy=simulate_greedy,
          observe=observe, forward=forward)
# warm up the jit so the first call is not slow
carenv.evaluate(_W, 4, seed=1); carenv.evaluate(_W, 1, seed=0, argmax=True)
if os.path.exists("/work/sim_calls.json"): _SIM["calls"] = json.load(open("/work/sim_calls.json"))["calls"]
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line); _SIM["turn_calls"] = 0
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(req["code"], "<model_code>", "exec"), NS)
    except Exception:
        buf.write(traceback.format_exc(limit=3))
    json.dump({"calls": _SIM["calls"]}, open("/work/sim_calls.json", "w"))
    _real_stdout.write(json.dumps({"out": buf.getvalue(), "sim_calls": _SIM["turn_calls"], "passes_used": _SIM["calls"]}) + "\n"); _real_stdout.flush()
