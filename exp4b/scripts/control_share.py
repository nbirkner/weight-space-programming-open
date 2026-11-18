"""How much of each successful episode is still driven by the trained network?
For each step before the episode ends, compare the edited network's entry-0 action distribution with the ORIGINAL
network's distribution on the same observation. 'Trained net in control' = total-variation distance < 0.1."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, wsp, carenv
base = wsp.shown_weights()[0]; bj = {k: jnp.asarray(v, jnp.float32) for k, v in base.items()}
def ep(params, rng):
    obs, state = carenv.ENV.reset(rng, carenv.EP, override_reset_state=carenv._env_state)
    def body(c, t):
        obs, state, done, rng = c
        rng, r1, r2 = jax.random.split(rng, 3)
        lg = carenv.forward(params, obs); act = carenv._sample(r1, lg, False)
        pe = jax.nn.softmax(lg[0:3]); pb = jax.nn.softmax(carenv.forward(bj, obs)[0:3])
        tv = 0.5 * jnp.abs(pe - pb).sum()
        o2, s2, rew, d2, info = carenv.ENV.step(r2, state, act, carenv.EP)
        alive = ~done
        obs = jnp.where(alive, o2, obs); state = jax.tree.map(lambda a, b: jnp.where(alive, a, b), s2, state)
        return (obs, state, done | d2, rng), (tv, alive)
    _, (tv, alive) = jax.lax.scan(body, (obs, state, jnp.array(False), rng), jnp.arange(256))
    return tv, alive
run = jax.jit(jax.vmap(ep, in_axes=(None, 0)))
out = {}
for name, f in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    d = np.load(f); p = {k: jnp.asarray(d[k], jnp.float32) for k in d.files}
    tv, alive = [np.asarray(x) for x in run(p, jax.random.split(jax.random.PRNGKey(5), 64))]
    share = float(((tv < 0.1) & alive).sum() / alive.sum()); steps = float(alive.sum(1).mean())
    out[name] = dict(trained_net_in_control_share=share, mean_episode_steps=steps)
    print(name, f"trained-net-in-control share {share:.2f}, mean episode length {steps:.1f} steps", flush=True)
json.dump(out, open(f"{wsp.ROOT}/results/control_share.json", "w"), indent=1)
