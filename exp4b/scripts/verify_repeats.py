"""Fresh-seed re-check of solved repeats, same as repair_flash_1's verification: 1,024 episodes on seed 424242 and seed 99
(reward-based success), plus 1,024 on seed 87654 scored by the environment's collision-based GoalR flag, plus greedy."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, wsp, carenv
base = wsp.shown_weights()[0]
def collision(params, seed=87654):
    P = {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}
    def episode(rng):
        obs, st = carenv.ENV.reset(rng, carenv.EP, override_reset_state=carenv._env_state)
        def step(c, t):
            o, s, done, success, rng = c
            rng, r1, r2 = jax.random.split(rng, 3)
            act = carenv._sample(r1, carenv.forward(P, o), False)
            o2, s2, rew, d, info = carenv.ENV.step(r2, s, act, carenv.EP)
            success = success | ((~done) & d & info['GoalR'])
            o = jnp.where(done, o, o2); s = jax.tree.map(lambda a, b: jnp.where(done, a, b), s, s2)
            return (o, s, done | d, success, rng), None
        (_, _, _, s, _), _ = jax.lax.scan(step, (obs, st, jnp.array(False), jnp.array(False), rng), None, length=256)
        return s
    run = jax.jit(jax.vmap(episode)); keys = jax.random.split(jax.random.PRNGKey(seed), 1024)
    return int(sum(int(np.asarray(run(keys[i*64:(i+1)*64])).sum()) for i in range(16)))
out = {}
for arg in sys.argv[1:]:
    name, f = arg.split("=")
    d = np.load(f); p = {k: d[k].astype(np.float64) for k in d.files}
    r1 = carenv.evaluate(p, 1024, seed=424242); r2 = carenv.evaluate(p, 1024, seed=99)
    c = collision(p); g = carenv.evaluate(p, 1, seed=0, argmax=True)["success"]
    out[name] = dict(file=f, seed_424242=r1["successes"], seed_99=r2["successes"], seed_87654_collision=c, greedy=g, changed=carenv.diff_count(base, p)["changed"])
    print(name, out[name], flush=True)
    json.dump(out, open(f"{wsp.ROOT}/results/verify_repeats.json", "w"), indent=1)
