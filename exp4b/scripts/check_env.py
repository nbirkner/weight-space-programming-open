import sys, time; sys.path.insert(0, "scripts")
from carenv import *
import paths
print("obs dim", ENV.observation_space(EP).shape if hasattr(ENV.observation_space(EP),'shape') else None)
b = load_base(); a = load_npz(paths.require_upstream('exp4_fix_car', 'networks', 'edited_actor.npz'))
t=time.time(); r = evaluate(b, 64, trace=True); print("base", {k:v for k,v in r.items() if k!='trace_ep0'}, time.time()-t); print(r["trace_ep0"])
t=time.time(); r = evaluate(a, 64, trace=True); print("astra", {k:v for k,v in r.items() if k!='trace_ep0'}, time.time()-t); print(r["trace_ep0"])
