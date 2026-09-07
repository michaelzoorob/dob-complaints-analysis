"""Launch a long-running command in its own session so it survives the parent shell.
Usage: python scripts/detach.py LOGFILE -- command args...
"""
import subprocess, sys
log, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
with open(log, "ab") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, cwd=None)
print(p.pid)
