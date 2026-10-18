"""Model client with a hard per-item call budget.

Every attempt -- including one that errors -- is counted, because the counting proxy at grading
time counts requests, not successes. The OpenAI client's own retries are switched OFF
(max_retries=0): a silent retry would break the exact-1x rule.
"""
import os
import re
import threading


_SECRET_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(\b(?:api[_-]?key|access[_-]?token)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)https?://[^\s/@:]+:[^\s/@]+@"), "https://[REDACTED]@"),
    # Provider errors can contain a key-management link whose opaque path
    # identifier is sensitive even though it is not the API secret itself.
    (re.compile(r"(?i)(https?://(?:www\.)?openrouter\.ai/workspaces/[^/\s]+/keys/)[A-Za-z0-9_-]+"),
     r"\1[REDACTED]"),
)
_SAFE_HEADERS = ("x-request-id", "request-id", "retry-after", "x-ratelimit-limit",
                 "x-ratelimit-remaining", "x-ratelimit-reset")
_SAFE_METADATA = ("error_type", "provider_code", "limit_source", "reason")


def _safe_text(value, limit=240):
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"(?i)\b(?:authorization|proxy-authorization)\s*[:=]\s*[^\s,;]+",
                  "[REDACTED_HEADER]", text)
    return text[:limit]


def _provider_fields(body):
    """Extract only known provider error fields; never retain an arbitrary body."""
    if not isinstance(body, dict):
        return {}
    err = body.get("error", body)
    if not isinstance(err, dict):
        return {}
    result = {}
    for key in ("code", "type"):
        if key in err:
            result["provider_" + key] = _safe_text(err[key], 100)
    if "message" in err:
        result["provider_message"] = _safe_text(err["message"], 240)
    metadata = err.get("metadata")
    if isinstance(metadata, dict):
        for key in _SAFE_METADATA:
            if key in metadata:
                result[key] = _safe_text(metadata[key], 120)
    return result


def exception_diagnostic(exc, item_id, stage="request"):
    """Return bounded, allowlisted failure metadata with secrets redacted."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None
    name = type(exc).__name__
    if status is not None:
        failure_stage = "http_response"
        category = "http_429" if status == 429 else "http_error"
    elif (isinstance(exc, OSError) or "timeout" in name.casefold()
          or "connection" in name.casefold() or "connect" in name.casefold()):
        failure_stage = "transport"
        category = "transport_error"
    elif "response" in name.casefold() or "json" in name.casefold():
        failure_stage = "response_decoding"
        category = "response_decoding_failure"
    else:
        failure_stage = stage
        category = "request_error"
    result = {"item_id": str(item_id), "stage": failure_stage, "category": category,
              "exception_class": name, "http_status": status}
    cause = exc
    seen = set()
    for _ in range(4):
        if id(cause) in seen:
            break
        seen.add(id(cause))
        errno = getattr(cause, "errno", None)
        if isinstance(errno, int):
            result["errno"] = errno
        nested = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
        if nested is None:
            break
        cause = nested
    if cause is not exc:
        result["cause_class"] = type(cause).__name__
    body = getattr(exc, "body", None)
    result.update(_provider_fields(body))
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        for key in _SAFE_HEADERS:
            value = headers.get(key)
            if value is None:
                value = next((v for k, v in headers.items() if str(k).casefold() == key), None)
            if value is not None:
                result[key.replace("-", "_")] = _safe_text(value, 120)
    if status is None:
        message_source = str(cause) if cause is not exc else str(exc)
        message = _safe_text(message_source)
        if message:
            result["message"] = message
    return result


class ModelCallFailure(RuntimeError):
    """A model request or response-decoding failure with safe metadata."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic.get("category", "model_call_failure"))


def _response_failure(item_id, category, exc=None):
    diagnostic = {"item_id": str(item_id), "stage": "response_decoding", "category": category,
                  "exception_class": type(exc).__name__ if exc is not None else None,
                  "http_status": None}
    if exc is not None:
        message = _safe_text(str(exc))
        if message:
            diagnostic["message"] = message
    return ModelCallFailure(diagnostic)


class BudgetExceeded(RuntimeError):
    pass


class ModelClient:
    """Thin wrapper over the OpenAI-compatible endpoint (OpenRouter, Granite 4.2 8B)."""

    def __init__(self, base_url, api_key, model, timeout=120.0):
        from openai import OpenAI
        self.model = model
        self._c = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

    def complete(self, item_id, messages, max_tokens):
        r = self._c.chat.completions.create(
            model=self.model, messages=messages,
            temperature=1.0, top_p=0.95, max_tokens=max_tokens,
            extra_headers={"X-Item-Id": item_id},
            extra_body={"reasoning": {"enabled": False}},
        )
        try:
            choices = r.choices
            content = choices[0].message.content if choices else None
        except Exception as e:
            raise _response_failure(item_id, "invalid_response_envelope", e) from None
        if not isinstance(content, str) or not content.strip():
            category = "empty_response_content" if content in (None, "") else "invalid_response_content"
            raise _response_failure(item_id, category)
        return content


def load_env(path=None):
    """Minimal .env reader (no dependency). Real environment variables win."""
    cands = [path] if path else [os.path.join(os.getcwd(), ".env"),
                                 os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                                     os.path.abspath(__file__)))), ".env")]
    for p in cands:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as env_file:
                for line in env_file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return p
    return None


def client_from_env(timeout=120.0):
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("MODEL") or "ibm-granite/granite-4.2-8b"
    if not key or not base:
        raise SystemExit("Missing endpoint config: set OPENROUTER_API_KEY and OPENROUTER_BASE_URL "
                         "(see .env.example).")
    if model != "ibm-granite/granite-4.2-8b":
        raise SystemExit("MODEL must be ibm-granite/granite-4.2-8b (got %r)." % model)
    return ModelClient(base, key, model, timeout)


class Budgeted:
    """Per-item gate: counts every attempt and refuses to exceed `cap`."""

    def __init__(self, client, item_id, cap, max_tokens):
        self.client, self.item_id, self.cap, self.max_tokens = client, item_id, cap, max_tokens
        self.used = 0
        self.responses_received = 0
        self.errors = []
        self._lock = threading.Lock()

    @property
    def left(self):
        return self.cap - self.used

    def ask(self, system, user):
        with self._lock:
            if self.used >= self.cap:
                raise BudgetExceeded("item %s: budget of %d exhausted" % (self.item_id, self.cap))
            self.used += 1
        try:
            result = self.client.complete(self.item_id, [{"role": "system", "content": system},
                                                         {"role": "user", "content": user}],
                                          self.max_tokens)
        except ModelCallFailure as e:
            if e.diagnostic.get("stage") == "response_decoding":
                # The endpoint returned something, but it was not a usable completion envelope.
                self.responses_received += 1
            diagnostic = e.diagnostic
        except Exception as e:            # counted above; never retried here
            diagnostic = exception_diagnostic(e, self.item_id)
        else:
            self.responses_received += 1
            if not isinstance(result, str) or not result.strip():
                diagnostic = {"item_id": str(self.item_id), "stage": "response_decoding",
                              "category": "empty_response_content", "exception_class": None,
                              "http_status": None}
            else:
                return result
        self.errors.append(diagnostic)
        raise ModelCallFailure(diagnostic)
