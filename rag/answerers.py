"""Answer backends that need no API key: a local Ollama model, and an extractive (no-LLM) fallback.

Every answerer exposes:
    name: str                     - shown in the UI / health endpoint
    has_credentials: bool         - True when it can run right now
    stream(messages) -> events    - {'type': 'text', 'text'} deltas, then one {'type': 'done', ...}
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from rag.llm import SYSTEM_PROMPT

# ---------------------------------------------------------------- Ollama (local LLM)


class OllamaError(RuntimeError):
    pass


def _opener(base_url: str) -> urllib.request.OpenerDirector:
    # Talk to a local Ollama directly, never through an HTTP proxy from the environment.
    host = urlparse(base_url).hostname or ""
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class OllamaAnswerer:
    """Streams answers from a model served by Ollama (https://ollama.com) on this machine."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434", timeout: float = 600):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self._opener = _opener(self.base_url)
        self._status: tuple[float, str | None] = (0.0, None)

    def _get_json(self, path: str, timeout: float = 2.0) -> Any:
        with self._opener.open(f"{self.base_url}{path}", timeout=timeout) as r:
            return json.loads(r.read())

    def check(self) -> str | None:
        """Return None if Ollama is up and the model is installed, else a human-readable problem."""
        try:
            tags = self._get_json("/api/tags")
        except (urllib.error.URLError, OSError, ValueError):
            return f"Ollama is not running at {self.base_url}. Install it from ollama.com and start it."
        names = {m.get("name", "") for m in tags.get("models", [])} | {m.get("model", "") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if self.model not in names and wanted not in names:
            return f"The Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}"
        return None

    def problem(self, max_age: float = 15.0) -> str | None:
        """check(), cached for a few seconds so the UI can poll cheaply."""
        ts, status = self._status
        if time.monotonic() - ts > max_age:
            status = self.check()
            self._status = (time.monotonic(), status)
        return status

    @property
    def has_credentials(self) -> bool:
        return self.problem() is None

    def stream(self, messages: list[dict]) -> Iterator[dict]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            resp = self._opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise OllamaError(f"Ollama error ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            self._status = (0.0, None)
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        final: dict[str, Any] = {}
        with resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise OllamaError(f"Ollama error: {chunk['error']}")
                text = (chunk.get("message") or {}).get("content", "")
                if text:
                    yield {"type": "text", "text": text}
                if chunk.get("done"):
                    final = chunk
                    break

        reason = final.get("done_reason", "stop")
        done: dict[str, Any] = {
            "type": "done",
            "stop_reason": "max_tokens" if reason == "length" else "end_turn",
            "model": self.name,
            "usage": {"input_tokens": final.get("prompt_eval_count"), "output_tokens": final.get("eval_count")},
        }
        if reason == "length":
            done["notice"] = "The answer was cut off at the model's length limit."
        yield done


# ---------------------------------------------------------------- Extractive (no LLM at all)

_STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves tell know please much many""".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_SOURCE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.S)


def _terms(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in _STOPWORDS or len(w) < 2:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)  # crude plural folding
    return out


class ExtractiveAnswerer:
    """Answers by quoting the most relevant sentences from the retrieved sources, with [n] citations.

    Needs no model, no download and no key. It cannot paraphrase or reason; it shows where the answer is.
    """

    name = "extractive (no LLM)"
    has_credentials = True

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    def answer(self, messages: list[dict]) -> str:
        content = messages[-1]["content"] if messages else ""
        question = content.rsplit("Question:", 1)[-1].strip()
        q_terms = set(_terms(question))
        sources = _SOURCE.findall(content)
        if not sources:
            return "I couldn't find anything about this in your documents. Try adding documents or rephrasing the question."

        # Candidate sentences, tagged with their source number.
        cands: list[tuple[int, int, str, list[str]]] = []  # (source n, position, sentence, terms)
        for n, text in sources:
            for pos, sent in enumerate(_SENT.split(text)):
                sent = sent.strip().lstrip("#").strip()
                if len(sent) < 15:
                    continue
                cands.append((int(n), pos, sent, _terms(sent)))
        if not cands or not q_terms:
            return "I couldn't find anything about this in your documents."

        df = Counter(t for *_, terms in cands for t in set(terms))
        total = len(cands)

        def score(terms: list[str]) -> float:
            hits = q_terms.intersection(terms)
            if not hits:
                return 0.0
            idf = sum(math.log(1 + total / df[t]) for t in hits)
            return idf * (len(hits) / len(q_terms)) / math.sqrt(1 + len(terms) / 25)

        ranked = sorted(((score(c[3]), c) for c in cands), key=lambda x: -x[0])
        best = ranked[0][0] if ranked else 0.0
        if best <= 0:
            return "I couldn't find anything about this in your documents."

        picked, seen = [], set()
        for s, (n, pos, sent, _) in ranked:
            if s < best * 0.35 or len(picked) >= self.max_sentences:
                break
            if sent.lower() in seen:
                continue
            seen.add(sent.lower())
            picked.append((n, pos, sent))

        lines = ["Here is what your documents say:", ""]
        lines += [f"- {sent} [{n}]" for n, pos, sent in picked]
        lines += ["", "*Quoted directly from your documents (no AI model is configured).*"]
        return "\n".join(lines)

    def stream(self, messages: list[dict]) -> Iterator[dict]:
        text = self.answer(messages)
        for piece in re.split(r"(?<=\n)", text):
            yield {"type": "text", "text": piece}
        yield {"type": "done", "stop_reason": "end_turn", "model": self.name, "usage": {}}
