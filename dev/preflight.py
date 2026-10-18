"""Pre-submission checks: no placeholders left, no secrets, entrypoint present, tests pass."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_key_material(root):
    """Scan distributable files, not the ignored local credential file."""
    leaks = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in (".git", ".venv", "runs", "__pycache__")]
        for f in files:
            if f in ("preflight.py", ".env"):
                continue
            try:
                with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as stream:
                    t = stream.read()
            except Exception:
                continue
            if re.search(r"sk-or-v1-[0-9a-f]{16,}|OPENROUTER_API_KEY=[ \t]*[A-Za-z0-9_\-]{12,}", t):
                leaks.append(os.path.relpath(os.path.join(d, f), root))
    return leaks


def main():
    checks = []

    def check(cond, msg):
        print(("PASS  " if cond else "FAIL  ") + msg)
        checks.append(bool(cond))

    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    check("FILL_ME" not in readme, "README has no FILL_ME placeholders (%d left)" % readme.count("FILL_ME"))
    check(os.access(os.path.join(ROOT, "run"), os.X_OK), "./run is executable")
    req = open(os.path.join(ROOT, "requirements.txt")).read()
    check(all("==" in l for l in req.split() if l), "requirements are pinned")
    check("Python" in readme, "README states the Python version")
    leaks = find_key_material(ROOT)
    check(not leaks, "no API key material in distributable files %s" % leaks)
    check(not os.path.exists(os.path.join(ROOT, ".env")) or ".env" in open(os.path.join(ROOT, ".gitignore")).read(),
          ".env is git-ignored")
    r = subprocess.run([sys.executable, "-I", "-m", "unittest", "discover", "-s", os.path.join(ROOT, "tests")],
                       capture_output=True, text=True, cwd=ROOT)
    check(r.returncode == 0, "unit tests pass")
    if r.returncode:
        print(r.stderr[-3000:])
    print("\nPREFLIGHT OK" if all(checks) else "\nPREFLIGHT FAILED")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
