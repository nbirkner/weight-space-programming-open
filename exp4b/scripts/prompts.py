import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as _paths
HEAD = open(f"{_paths.ROOT}/results/prompt_head.txt").read()
OBS = open(f"{_paths.ROOT}/results/obs_summary.txt").read()
# The scene / rules / network sections of the paper's own prompt, without the paper's "no tools" task paragraph,
# the edit-format section and the weight dump.
_s = HEAD.index("THE SCENE AND THE RULES"); _e = HEAD.index("EDIT FORMAT")
SCENE = HEAD[_s:_e].strip()
import json
BASE_RESULT = json.load(open(_paths.require_upstream("exp4_fix_car", "eval", "base_result.json")))["text"]

def tools_prompt(pass_budget, max_turns, max_changed=0):
    RESULT = BASE_RESULT
    LIMIT = (f"\nREPAIR LIMIT FOR THIS RUN: this is a repair, not a rewrite. Keep the trained network and add or change only what is needed: at most {max_changed:,} of the 171,664 weights may differ from the shown network (save_edit refuses more). Do not zero out or replace whole layers; the trained network's own units should still be doing their job in the parts where it already behaves well.\n" if max_changed else "")
    return f"""Below is a policy network that controls a small car in a 2-D physics scene, and the scene. The network was trained with a standard reinforcement-learning algorithm (PPO) for 1,000,000,000 environment steps, and it still never succeeds.

YOUR TASK: change the network's weights so that the car succeeds as often as possible. First inspect the weights and explain why the trained network fails. Then keep the parts of the trained network that already work and fix the part that causes the failure. What matters most is that the car succeeds: do not give up success to change fewer weights. Your edited network is measured on 1,024 new episodes with actions sampled from its policy.

HOW THIS RUN WORKS (this is different from a no-tools version of the task)
You have a persistent Python kernel (numpy available; no internet; CPU only). Each of your replies may contain ONE ```python code block. I run the last ```python block in your reply and send you back what it printed (stdout, truncated to 8,000 characters). Do not write the output yourself; wait for it. When you are finished, reply WITHOUT a python block; that ends the run and your last saved edit is scored.
At most {max_turns} replies in total. Each code run has a 240 s time limit.

Already defined in the kernel:
- W: dict of the network's tensors exactly as you would edit them (numpy float64): 'embed.W' (679,128), 'embed.b', 'actor0.W'..'actor4.W' (128,128), 'actor0.b'..'actor4.b', 'actor_head.W' (128,16), 'actor_head.b' (16). Treat W as read-only; use W_copy() to get a fresh copy to edit.
- forward(params, obs) -> logits (16,) or (N,16) for observation vector(s) of length 679.
- observe(params=None, actions=[...], n_steps=64) -> (obs array (T,679), entry-0 logits (T,3)). With a fixed list of action-entry-0 values (0 off, 1 drive +1, 2 drive -1) it rolls the scene out deterministically and is free. With actions=None the network drives (greedy) and it costs 1 simulation pass.
- simulate(params, n=64, trace=True) -> dict with success rate, red / timeout shares, mean return and (trace=True) one sampled episode's positions every 16 steps. n <= 256. Costs 1 simulation pass. At most 4 simulation calls per code run.
- simulate_greedy(params) -> the most-likely-action episode (1 pass).
- apply_edits(text, params=None) -> (new_params, report): applies statements in the edit language below (e.g. "actor0.W[6, 122] = 0.8") to a copy of params (default: W).
- save_edit(params): saves params as your edited network. This is what gets scored. Call it whenever you have a better network.
- passes_left(): simulation passes left. You have {pass_budget} passes for the whole run.

Edit language (for apply_edits): one statement per line, e.g. actor_head.b[0:3] = [0.5, 2.0, -1.0]; actor_head.W[7, 0:3] = 0; embed.W[26, 5] = 1.25; actor0.W[:, 3] *= 0.5 (also += and -=); embed.b[5] -= 0.1; ZERO actor4.W. Indices are half-open; ':' alone = the whole axis. You can also edit numpy arrays directly in code.

RULES: You are programming the weights yourself. Do NOT use gradient descent, autograd, optimizers, evolutionary / random / grid search over weight values, behaviour cloning or any other training or fitting of weights to a loss or to simulation results. Use code to read the weights, compute activations, work out by reasoning what values to write, and test your design with the limited simulation passes. Write weights you chose and can explain. Keep each reply short: a few sentences of reasoning, then the code.
{LIMIT}
{SCENE}

COMPRESSED VIEW OF THE OBSERVATION (computed by us from rollouts of this fixed scene)
{OBS}

THE CURRENT NETWORK'S RESULT
{RESULT}

Start by looking at the weights and the network's behaviour with code."""

FEEDBACK = """Your saved edit was scored by us on 256 episodes (actions sampled from its policy): {res}
One sampled episode, every 16 steps:
{trace}
{extra}You may continue (same kernel, same rules, {passes} simulation passes left, {turns} replies left). Reply without a python block when you are finished."""
