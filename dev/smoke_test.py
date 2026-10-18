"""One production extraction/solver smoke test on the first real dataset item."""
import argparse
import json
import os
import stat
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from auric.client import client_from_env, load_env  # noqa: E402
from auric.pipeline import ItemRunFailure, answer_item  # noqa: E402


def failure_summary(meta):
    """Expose bounded diagnostics, never model text, prompts, or operands."""
    parses = []
    for p in meta.get("parses", []):
        invalid = p.get("invalid", {})
        parses.append({"source": p.get("source"),
                       "valid_record_count": len(p.get("readings", {})),
                       "invalid_indices": sorted(invalid.keys(), key=str),
                       "invalid_categories": {str(i): _invalid_category(reason)
                                              for i, reason in invalid.items()}})
    return {"item_id": meta.get("id"), "calls": meta.get("calls"),
            "failure": meta.get("failure"), "failure_stage": meta.get("failure_stage"),
            "unresolved_line_indices": meta.get("unresolved_line_indices", []),
            "outcome": meta.get("outcome"), "diagnostic": meta.get("diagnostic"),
            "errors": meta.get("errors", []),
            "parses": parses}


def _invalid_category(reason):
    if reason == "argument not in source line":
        return "source_grounding"
    if reason.startswith("bad person/") or reason.startswith("bad person/time"):
        return "operand_type_or_membership"
    if reason.startswith("unknown name"):
        return "unsupported_operand"
    if reason.startswith("unknown relation"):
        return "unsupported_relation"
    if reason == "conflicting duplicate outputs":
        return "duplicate_conflict"
    return "invalid_record"


def write_protected_capture(path, meta):
    """Store only returned completion text in a new owner-only file, if requested."""
    chunks = [r["text"] for r in meta.get("raw", [])
              if isinstance(r, dict) and isinstance(r.get("text"), str)]
    if not chunks:
        return False
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if stat.S_IMODE(os.stat(parent).st_mode) & 0o077:
        raise PermissionError("capture directory must be owner-only")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True, help="assignment input JSON path")
    ap.add_argument("--item-index", type=int, default=0, help="zero-based item position (default: first)")
    ap.add_argument("--capture-response", default="",
                    help="optional private path for raw completion text (new owner-only file only)")
    args = ap.parse_args(argv)
    if args.item_index < 0:
        ap.error("--item-index must be non-negative")
    load_env(os.path.join(ROOT, ".env"))
    c = client_from_env()
    with open(args.items, encoding="utf-8") as stream:
        items = json.load(stream)
    if args.item_index >= len(items):
        ap.error("--item-index is outside the input item list")
    item = items[args.item_index]
    try:
        answer, meta = answer_item(item, c, 1)
    except ItemRunFailure as e:
        captured = write_protected_capture(args.capture_response, e.meta) if args.capture_response else False
        summary = failure_summary(e.meta)
        if captured:
            summary["protected_capture"] = args.capture_response
        print("FAILED: " + json.dumps(summary, sort_keys=True))
        return 1
    captured = write_protected_capture(args.capture_response, meta) if args.capture_response else False
    print("OK: item=%s model=%s calls=%s case=%s" %
      (item["id"], c.model, meta.get("calls"), answer["case"]))
    if captured:
        print("protected response capture saved: " + args.capture_response)
    if meta.get("outcome"):
        print("solver: worlds=%s classification=%s" %
              (meta["outcome"].get("n"), meta["outcome"].get("kind")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
