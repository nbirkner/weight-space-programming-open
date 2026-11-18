# Open-model replication of Weight-Space Programming

An attempt to replicate [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) with open models on Together. The paper shows that a frontier closed model can edit a neural network's weights in one shot. This repo is the code, run logs, and results from trying the same tasks with cheaper open models, including a strict one-shot test and a version where the model can use a simulator.

This is not a claim that open models match the paper's unaided one-shot result. On the car task they did not. With tools, a cheap open model could write a working edit about 4 times in 10, at about $0.14 per attempt.

## Credit

The paper and the released logs and networks are by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, MIT CSAIL.

- Paper: https://rikiyat.github.io/blog/weight-space-programming/
- Released logs and networks: https://github.com/RikiyaT/lm_understand-logs

This repo does not vendor that upstream tree. Clone it separately and point `WSP_UPSTREAM` at it (see Setup).

## How this was made

The runs themselves were done by AI research agents on my setup. I directed the work, decided what to test and checked what came back.

## Setup

Python 3.11, with:

- jax 0.9.0
- jaxlib 0.9.0
- kinetix-env 3.0.2
- flax 0.12.6
- numpy 2.4.6

```bash
git clone https://github.com/RikiyaT/lm_understand-logs
python3.11 -m venv exp4b/.venv
source exp4b/.venv/bin/activate
pip install -r requirements.txt
export WSP_UPSTREAM=/path/to/lm_understand-logs
```

If the virtualenv is not at `exp4b/.venv`, set `WSP_VENV` to it.

Paid runs call the Together API. Set `TOGETHER_API_KEY` in the environment. Do not put the key in a file in this repo. Set a spend limit on the Together account first. The ledger in `exp4b/ledger.json` will refuse new calls once `WSP_LEDGER_HARD_STOP` is reached (default $9.50). Each assisted attempt cost about $0.14.

## How to run

Commands below assume you are in `exp4b/` with the virtualenv active and `WSP_UPSTREAM` set. The model used for the assisted car runs was `deepseek-ai/DeepSeek-V4.1-Flash`.

**Strict one-shot test** (the paper's prompt, no tools, one reply):

```bash
python scripts/run_raw.py --model deepseek-ai/DeepSeek-V4.1-Flash --tag my_strict --rounds 1 --max_tokens 32768
```

**Assisted run** (tools, simulator, feedback, 8,600-weight cap — the winning setup):

```bash
python scripts/run_tools.py --model deepseek-ai/DeepSeek-V4.1-Flash --tag my_repair \
  --max_turns 30 --passes 30 --max_tokens 16384 --feedback_rounds 3 \
  --run_cap 0.15 --max_changed 8600 --wall 600 --seed 201
```

A free dry run, with no API calls, applies the paper's released edit through the same tool path:

```bash
python scripts/run_tools.py --model deepseek-ai/DeepSeek-V4.1-Flash --tag mock_dryrun \
  --mock scripts/mock_model.py --max_turns 3 --passes 5 --feedback_rounds 0
```

**Controls** (no language model, $0):

```bash
python scripts/search_control.py --method random --seed 1 --budget 120
python scripts/search_control.py --method greedy --seed 3 --budget 120
```

**Leakage check** (same assisted setup on a shuffled copy of the network, so unit indices no longer match the paper's edit). `networks/perm_shown.npz` is already in this repo; rebuild it with `python scripts/make_perm.py` if you want to regenerate it.

```bash
python scripts/run_tools_perm.py --model deepseek-ai/DeepSeek-V4.1-Flash --tag my_leak \
  --max_turns 30 --passes 30 --max_tokens 16384 --feedback_rounds 3 \
  --run_cap 0.15 --max_changed 8600 --wall 600 --seed 301
```

**Check a saved network** on fresh seeds (1024 episodes each):

```bash
python scripts/verify.py repair_flash_1=runs/repair_flash_1/submission_t13.npz
```

The base (unedited, shown) network lives at `networks/base_shown.npz` and scores 0/1024.

Exp 1c (which game is this network?) has its own runner in `exp1c/scripts/run_exp1c.py`. Prompts are already built under `exp1c/prompts/`. Rebuilding them also needs `WSP_UPSTREAM` (`python scripts/build_prompts.py` from `exp1c/`).

## Results

Car task (Exp 4b):

- Strict one-shot: 0/1024 in 7 runs, across 7 Together model IDs.
- Assisted (tools + simulator + feedback, 8,600-weight cap): 4 of 10 repeat runs scored 1024/1024. Median cost was about $0.14 per attempt, or about $0.35 per success.
- Random search with no model: 0 of 4 seeds.
- Greedy unit-rewire search with no model: 1 of 6 seeds reached 1,023/1024.
- Leakage check on the shuffled network: 3 of 3 runs scored 1024/1024, and none used the permuted images of the paper's units 126 and 127.

The first working assisted edit (`runs/repair_flash_1`) also scored 1024/1024 and changed 1.7% of weights. It is a blunt fix: it takes over two hidden units and rewires the motor. A simple search found the same kind of edit once, so this does not show that language models are uniquely good at the repair.

Full write-ups, cost tables, and per-run logs:

- `exp4b/reports/RESULTS.md` and `exp4b/reports/COSTS.md`
- `exp1c/reports/RESULTS.md`
- `exp4b/runs/` and `exp4b/control/`

Scratch dry-runs and killed orphan runs were omitted from this copy; they are noted in the results write-up and were not counted as attempts.
