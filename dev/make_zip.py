"""Build a minimal, allowlisted submission ZIP without visible data or secrets."""
import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "auric-r1-submission.zip")
ROOT_FILES = {
    ".env.example", ".gitignore", "README.md", "EVALUATION_REPORT.md",
    "ABLATION_TABLE.md", "SUBMISSION_CHECKLIST.md",
    "RUN_AND_TEST.md", "requirements.txt", "run",
}
DEV_FILES = {
    "__init__.py", "completeness.py", "edits.py", "make_zip.py",
    "mock_model.py", "preflight.py", "replay_cached.py", "run_experiments.py", "smoke_test.py",
}


def build(out=DEFAULT_OUT):
    out = os.path.abspath(out)
    members = []
    for name in sorted(ROOT_FILES):
        members.append((name, name))
    brief_path = ("data/CANDIDATE_BRIEF.md"
                  if os.path.isfile(os.path.join(ROOT, "data", "CANDIDATE_BRIEF.md"))
                  else "CANDIDATE_BRIEF.md")
    members.append((brief_path, "CANDIDATE_BRIEF.md"))
    for source_dir, archive_dir in (("src/auric", "src/auric"), ("tests", "tests")):
        for current, dirs, files in os.walk(os.path.join(ROOT, source_dir)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for filename in sorted(files):
                if filename.endswith((".pyc", ".zip")):
                    continue
                absolute = os.path.join(current, filename)
                rel = os.path.relpath(absolute, ROOT).replace(os.sep, "/")
                members.append((rel, rel))
    for filename in sorted(DEV_FILES):
        rel = "dev/" + filename
        if os.path.isfile(os.path.join(ROOT, rel)):
            members.append((rel, rel))

    # Create at the requested path only after the allowlist is resolved. Callers
    # use a new name; mode bits from ZipInfo preserve the executable launcher.
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as z:
        for rel, archive_rel in members:
            p = os.path.join(ROOT, rel)
            info = zipfile.ZipInfo.from_file(p, os.path.join("auric-r1", archive_rel))
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(p, "rb") as stream:
                z.writestr(info, stream.read())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    print("wrote", build(args.out))
