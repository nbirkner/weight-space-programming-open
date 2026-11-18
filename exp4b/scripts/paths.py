"""Resolve repo paths without hard-coded machine locations.

Set WSP_UPSTREAM to a clone of https://github.com/RikiyaT/lm_understand-logs
(the paper's released logs and networks). Set WSP_VENV if the Python
environment is not at exp4b/.venv. Set WSP_ROOT only if this package was moved.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("WSP_ROOT", os.path.join(SCRIPT_DIR, "..")))


def _upstream_candidates():
    repo_root = os.path.dirname(ROOT)
    cands = []
    env = os.environ.get("WSP_UPSTREAM")
    if env:
        cands.append(env)
    cands.extend(
        [
            os.path.join(repo_root, "lm_understand-logs"),
            os.path.join(ROOT, "lm_understand-logs"),
            os.path.join(repo_root, "..", "lm_understand-logs"),
        ]
    )
    return [os.path.abspath(c) for c in cands]


def find_upstream():
    for c in _upstream_candidates():
        if os.path.isdir(os.path.join(c, "exp4_fix_car")):
            return c
    return _upstream_candidates()[0]


UPSTREAM = find_upstream()


def require_upstream(*rel):
    path = os.path.join(UPSTREAM, *rel)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing %s. Clone https://github.com/RikiyaT/lm_understand-logs "
            "and set WSP_UPSTREAM to that directory." % path
        )
    return path


PROMPT = os.path.join(UPSTREAM, "exp4_fix_car", "runs", "oneshot_repair", "reply", "prompt.txt")
ASTRA_ANSWER = os.path.join(UPSTREAM, "exp4_fix_car", "runs", "oneshot_repair", "reply", "answer.md")
INSTRUCTIONS = os.path.join(UPSTREAM, "exp4_fix_car", "runs", "oneshot_repair", "instructions.md")
PPO_BASE = os.path.join(UPSTREAM, "exp4_fix_car", "networks", "ppo_final_seed0.npz")
ASTRA_EDIT = os.path.join(UPSTREAM, "exp4_fix_car", "networks", "edited_actor.npz")
BASE_RESULT_JSON = os.path.join(UPSTREAM, "exp4_fix_car", "eval", "base_result.json")


def venv_dir():
    if os.environ.get("WSP_VENV"):
        return os.path.abspath(os.environ["WSP_VENV"])
    return os.path.join(ROOT, ".venv")


def python_bin():
    exe = os.path.join(venv_dir(), "bin", "python")
    return exe if os.path.exists(exe) else sys.executable


def venv_bwrap_binds():
    """Bind the venv and any Python install the venv interpreter points at.

    Follows the symlink chain so versioned installs (for example a uv
    python tree) are visible at the path the venv binary actually uses.
    """
    venv = venv_dir()
    venv_real = os.path.realpath(venv)
    binds = [("--ro-bind", venv, venv)]
    seen = {venv_real}

    def add_prefix(prefix):
        prefix = os.path.abspath(prefix)
        if prefix in seen:
            return
        if prefix == venv_real or prefix.startswith(venv_real + os.sep):
            return
        seen.add(prefix)
        binds.append(("--ro-bind", prefix, prefix))

    path = os.path.join(venv, "bin", "python")
    visited = set()
    while path and os.path.lexists(path):
        path = os.path.abspath(path)
        if path in visited:
            break
        visited.add(path)
        add_prefix(os.path.dirname(os.path.dirname(path)))
        marker = os.sep + "uv" + os.sep + "python" + os.sep
        if marker in path:
            add_prefix(path[: path.index(marker) + len(os.sep + "uv" + os.sep + "python")])
        if os.path.islink(path):
            target = os.readlink(path)
            path = target if os.path.isabs(target) else os.path.join(os.path.dirname(path), target)
        else:
            add_prefix(os.path.dirname(os.path.dirname(os.path.realpath(path))))
            break
    return venv, binds
