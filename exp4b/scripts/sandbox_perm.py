"""Parent side of the sandbox: bubblewrap, no network, cleared env, persistent worker, per-run timeout."""
import subprocess, json, os, select, time
from paths import ROOT, venv_bwrap_binds
class Sandbox:
    def __init__(self, workdir, timeout=240, pass_budget=30, max_changed=0):
        self.pass_budget = pass_budget; self.max_changed = max_changed
        self.workdir = workdir; os.makedirs(workdir, exist_ok=True); self.timeout = timeout; self.p = None; self.start()
    def start(self):
        venv, pybinds = venv_bwrap_binds()
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--clearenv",
               "--setenv", "HOME", "/work", "--setenv", "MPLCONFIGDIR", "/tmp", "--setenv", "JAX_PLATFORMS", "cpu",
               "--setenv", "TF_CPP_MIN_LOG_LEVEL", "3", "--setenv", "WSP_ROOT", "/exp4b", "--setenv", "PASS_BUDGET", str(self.pass_budget), "--setenv", "MAX_CHANGED", str(self.max_changed),
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
               "--ro-bind-try", "/bin", "/bin", "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
               * [x for bind in pybinds for x in bind], "--ro-bind", f"{ROOT}/scripts", "/exp4b/scripts",
               "--ro-bind", f"{ROOT}/results/models.json", "/exp4b/results/models.json",
               "--ro-bind", f"{ROOT}/networks/perm_shown.npz", "/exp4b/shown.npz",
               "--bind", self.workdir, "/work", "--chdir", "/work", "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
               f"{venv}/bin/python", "-u", "/exp4b/scripts/sandbox_worker_perm.py"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(f"{self.workdir}/worker.stderr", "a"), text=True, bufsize=1)
        line = self._readline(300)
        if not line or "ready" not in line: raise RuntimeError("sandbox failed to start: " + str(line))
    def _readline(self, t):
        r, _, _ = select.select([self.p.stdout], [], [], t)
        return self.p.stdout.readline() if r else None
    def run(self, code):
        t0 = time.time()
        self.p.stdin.write(json.dumps({"code": code}) + "\n"); self.p.stdin.flush()
        line = self._readline(self.timeout)
        if line is None:
            self.p.kill(); self.start()
            return {"out": f"[timed out after {self.timeout}s; the python state was reset]", "sim_calls": None, "secs": round(time.time() - t0, 1)}
        d = json.loads(line); d["secs"] = round(time.time() - t0, 1); return d
    def close(self):
        try: self.p.kill()
        except Exception: pass
