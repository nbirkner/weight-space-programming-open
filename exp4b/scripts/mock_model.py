"""Mock model for a free dry run. Turn 1 writes code followed by a FAKE output fence (the old bug's trigger).
Turn 2 applies the paper's released Astra edit via apply_edits and save_edit (to prove the scoring path end to end)."""
import re
import paths
ASTRA = re.findall(r"```edits[^\n]*\n(.*?)```", open(paths.require_upstream('exp4_fix_car', 'runs', 'oneshot_repair', 'reply', 'answer.md')).read(), re.S)[-1]
def reply(messages):
    n = sum(1 for m in messages if m["role"] == "assistant")
    if n == 0:
        c = ("Let me look.\n```python\nprint('MOCK_EXEC_OK', W['actor_head.b'][:3], passes_left())\n"
             "o, lg = observe(actions=[2]*20, n_steps=20); print('body x after 20 reverse steps', round(float(o[-1][52]),3))\n```\n"
             "```output\nFAKE OUTPUT THAT MUST NOT BE USED\n```\n")
    elif n == 1:
        c = "Apply a known edit.\n```python\np, rep = apply_edits('''" + ASTRA + "''')\nprint(rep['applied'], rep['errors'][:2])\nsave_edit(p)\nprint(simulate(p, 32, trace=False))\n```"
    else:
        c = "Done. The saved edit is my answer."
    return dict(content=c, reasoning="", usage=dict(prompt_tokens=0, completion_tokens=0), finish="stop", error=None, wall=0.0, estimated=False, ledger_total=0.0)
