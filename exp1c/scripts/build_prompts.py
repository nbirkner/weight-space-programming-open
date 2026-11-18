"""Rebuild Exp 1c prompts from the released networks (upstream repo).

Rebuilt 2026-11-16 from the Notion write-ups; the original Exp 1b builder lived on
another computer. Formats follow the Notion descriptions, not the original bytes.
"""
import json, os, re
import numpy as np

EXP1C = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.dirname(EXP1C)
def _upstream():
    cands = []
    if os.environ.get("WSP_UPSTREAM"):
        cands.append(os.environ["WSP_UPSTREAM"])
    cands.extend([os.path.join(REPO, "lm_understand-logs"), os.path.join(REPO, "..", "lm_understand-logs")])
    for c in cands:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, "exp1_which_game")):
            return c
    return os.path.abspath(cands[0])
UP = os.path.join(_upstream(), "exp1_which_game")
OUT = os.path.join(EXP1C, "prompts")
NETS = {"breakout": "a9f535cc", "pong": "dfc6cb65", "pacman": "bdf99279", "freeway": "0bbcc3ea"}
CH_NAMES = ["agent", "moving objects", "reward targets", "obstacles", "previous frame"]
GLYPHS = " .:-=+*#%@"  # 10 levels, scaled per channel from 0 to the channel max

def load(nid, shuffle_seed=None):
    d = np.load(f"{UP}/networks/{nid}.npz")
    W1 = np.round(d["actor/dense1/kernel"], 3); b1 = np.round(d["actor/dense1/bias"], 3)
    W2 = np.round(d["actor/dense2/kernel"], 3); b2 = np.round(d["actor/dense2/bias"], 3)
    W3 = np.round(d["actor/logits/kernel"], 3); b3 = np.round(d["actor/logits/bias"], 3)
    if shuffle_seed is not None:
        # Decoy: permute the 500 input rows of W1, destroying spatial and channel structure.
        perm = np.random.default_rng(shuffle_seed).permutation(500)
        W1 = W1[perm]
    return W1, b1, W2, b2, W3, b3

def prefix(game, anon=False):
    t = open(f"{UP}/runs/{game}/prompt.txt").read()
    p = t[: t.find("THE NETWORK'S WEIGHTS")]
    if anon:
        # Strip channel meanings. Keep indices, grid, actions and network formula.
        p = re.sub(r"  channel 0  agent: the cells the player controls", "  channel 0", p)
        p = re.sub(r"  channel 1  moving objects: things other than the agent that move", "  channel 1", p)
        p = re.sub(r"  channel 2  reward targets: things that give reward", "  channel 2", p)
        p = re.sub(r"  channel 3  obstacles: cells the agent cannot enter or that end the episode \(the wall border is always here\)", "  channel 3", p)
        p = re.sub(r"  channel 4  previous frame: the cells that were in channel 0 or channel 1 on the previous step", "  channel 4", p)
        p = p.replace("Each cell has 5 binary channels (0 or 1):", "Each cell has 5 binary channels (0 or 1). What each channel means is not given:")
        assert "agent:" not in p and "previous frame" not in p and "obstacles" not in p, p
    return p

def fmt(v): return f"{v:.3f}"
def mat(M): return "\n".join(" ".join(fmt(x) for x in row) for row in M)

def l2maps(W1):
    # L2 norm of each input's 32 outgoing weights, arranged as 5 channels x 10 x 10.
    n = np.sqrt((W1 ** 2).sum(1))  # 500
    return n.reshape(10, 10, 5).transpose(2, 0, 1)

def glyph_grid(m):
    mx = m.max()
    rows = []
    for r in range(10):
        rows.append(f"{r} |" + "".join(GLYPHS[min(9, int(v / mx * 10))] * 2 for v in m[r]) + "|")
    return "   " + "".join(f"{c} " for c in range(10)) + "\n" + "\n".join(rows)

def num_grid(m):
    return "\n".join(f"{r}: " + " ".join(f"{v:.2f}" for v in m[r]) for r in range(10))

def chlabel(c, anon): return f"channel {c}" if anon else f"channel {c} ({CH_NAMES[c]})"

def topk(W1, k=8, anon=False):
    lines = []
    for j in range(32):
        idx = np.argsort(-np.abs(W1[:, j]))[:k]
        parts = []
        for i in idx:
            cell, ch = divmod(int(i), 5); r, c = divmod(cell, 10)
            parts.append(f"(r{r},c{c},ch{ch}) {fmt(W1[i, j])}")
        lines.append(f"unit {j}: " + "; ".join(parts))
    return "\n".join(lines)

def strongest_cells(m, n=5):
    idx = np.argsort(-m.ravel())[:n]
    return ", ".join(f"(r{i // 10},c{i % 10}) {m.ravel()[i]:.2f}" for i in idx)

HEAD = ("THE NETWORK'S WEIGHTS (SUMMARISED)\n"
        "Policy network weights. Forward pass: h1 = tanh(x @ W1 + b1); h2 = tanh(h1 @ W2 + b2); logits = h2 @ W3 + b3. "
        "Values are computed from weights rounded to 3 decimals.\n"
        "W1 is shown as maps: for every input cell (row, column) and channel, the L2 norm of that input's 32 outgoing W1 weights. "
        "Glyph maps use the scale \"" + GLYPHS + "\" (blank = 0, @ = channel maximum), two characters per cell.\n")

def later_layers(b1, W2, b2, W3, b3):
    return ("\nb1 (layer 1 biases), shape 32:\n" + " ".join(fmt(x) for x in b1) +
            "\n\nW2 (layer 2 weights), shape 32 x 32: one line per row\n" + mat(W2) +
            "\n\nb2 (layer 2 biases), shape 32:\n" + " ".join(fmt(x) for x in b2) +
            "\n\nW3 (output layer weights), shape 32 x 5: one line per row\n" + mat(W3) +
            "\n\nb3 (output layer biases), shape 5:\n" + " ".join(fmt(x) for x in b3) + "\n")

OMIT = "\nOnly a summary of the first layer is shown. b1, W2, b2, W3 and b3 are omitted.\n"

def body(cond, W1, b1, W2, b2, W3, b3, anon=False):
    maps = l2maps(W1)
    s = HEAD
    if cond in ("heatmap", "w1_only", "ascii"):
        for c in range(5):
            s += f"\nW1 L2 map, {chlabel(c, anon)}, max {maps[c].max():.2f}:\n" + glyph_grid(maps[c]) + "\nValues:\n" + num_grid(maps[c]) + "\n"
    if cond == "glyphs":
        for c in range(5):
            m = maps[c]
            s += (f"\nW1 L2 map, {chlabel(c, anon)}: min {m.min():.2f}, max {m.max():.2f}, mean {m.mean():.2f}; "
                  f"strongest cells {strongest_cells(m)}\n" + glyph_grid(m) + "\n")
    if cond in ("heatmap", "w1_only", "topk8"):
        s += "\nTop-8 W1 connections per hidden unit, by absolute weight: (row, column, channel) weight\n" + topk(W1, anon=anon) + "\n"
    if cond == "heatmap":
        s += later_layers(b1, W2, b2, W3, b3)
    else:
        s += OMIT
    return s

def build():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    def write(name, game, nid, cond, text, truth):
        fn = f"{OUT}/{cond}/{game}_{name}.txt"; os.makedirs(os.path.dirname(fn), exist_ok=True)
        open(fn, "w").write(text)
        manifest.append(dict(file=fn, game=game, net=nid, cond=cond, truth=truth, chars=len(text)))
    for game, nid in NETS.items():
        W = load(nid)
        for cond in ("heatmap", "w1_only", "ascii", "glyphs", "topk8"):
            write(nid, game, nid, cond, prefix(game) + body(cond, *W), game)
        write(nid, game, nid, "heatmap_anon", prefix(game, anon=True) + body("heatmap", *W, anon=True), game)
        for sd in (101, 102):
            Ws = load(nid, shuffle_seed=sd)
            write(f"{nid}_shuf{sd}", game, nid, "heatmap_shuffled", prefix(game) + body("heatmap", *Ws), "decoy")
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    return manifest

if __name__ == "__main__":
    m = build()
    from collections import defaultdict
    agg = defaultdict(list)
    for r in m: agg[r["cond"]].append(r["chars"])
    for k, v in agg.items(): print(k, len(v), "files, mean chars", int(np.mean(v)))
