"""Independent success check: use the environment's collision-based GoalR flag, not reward thresholds.
Evaluate 1,024 fresh episodes in 16 batches of 64; compare to the earlier reward-based evaluation."""
import sys,json;sys.path.insert(0,'scripts')
import numpy as np,jax,jax.numpy as jnp,carenv
D=np.load('runs/repair_flash_1/submission_t13.npz'); P={k:jnp.asarray(D[k],jnp.float32) for k in D.files}
def episode(rng):
 obs,st=carenv.ENV.reset(rng,carenv.EP,override_reset_state=carenv._env_state)
 def step(c,t):
  o,s,done,success,rng=c
  rng,r1,r2=jax.random.split(rng,3)
  act=carenv._sample(r1,carenv.forward(P,o),False)
  o2,s2,rew,d,info=carenv.ENV.step(r2,s,act,carenv.EP)
  success=success|((~done)&d&info['GoalR'])
  o=jnp.where(done,o,o2);s=jax.tree.map(lambda a,b:jnp.where(done,a,b),s,s2)
  return (o,s,done|d,success,rng),None
 (_,_,_,s,_),_=jax.lax.scan(step,(obs,st,jnp.array(False),jnp.array(False),rng),None,length=256)
 return s
run=jax.jit(jax.vmap(episode))
keys=jax.random.split(jax.random.PRNGKey(87654),1024)
n=0
for i in range(16):
 a=np.asarray(run(keys[i*64:(i+1)*64]));n+=int(a.sum());print(i,n,flush=True)
r=dict(successes=n,episodes=1024,seed=87654,scoring='environment collision-based info GoalR at first done, not reward threshold')
json.dump(r,open('results/verify_collision.json','w'),indent=2);print(r,flush=True)
