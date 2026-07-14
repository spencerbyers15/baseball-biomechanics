"""LLM client abstraction.

Two implementations:
- AnthropicClient: real API calls with exponential-backoff retry.
- MockLLMClient: deterministic, offline. Prompts embed a machine-readable
  ```json fenced block (TASK_ID + inputs) which the mock parses to produce
  schema-valid outputs, so the entire harness can be exercised without a key.

Every call is logged as one JSONL row: prompt sha256, model, token counts,
latency, task id.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "claude-sonnet-4-6"

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


def _mock_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def approx_tokens(text: str) -> int:
    """Cheap token estimate (chars / 4), used for budgets and mock logging."""
    return max(1, len(text) // 4)


def extract_json(text: str) -> Any:
    """Parse the first JSON object in a model response (fenced or bare)."""
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            raise ValueError(f"no JSON object found in response: {text[:200]!r}")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    break
        if candidate is None:
            raise ValueError("unbalanced JSON object in response")
    return json.loads(candidate)


class CallLogger:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict) -> None:
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


class LLMClient:
    """Base client. Subclasses implement _complete(prompt, system, ...).

    Transient failures (per _should_retry) are retried with exponential
    backoff (2s, 4s, 8s, ...) up to max_retries times.
    """

    def __init__(self, model: str = DEFAULT_MODEL, log_path: Path | None = None,
                 max_retries: int = 4):
        self.model = model
        self.logger = CallLogger(log_path)
        self.max_retries = max_retries
        self._sleep = time.sleep  # injectable for tests

    def _should_retry(self, exc: Exception) -> bool:
        return isinstance(exc, (ConnectionError, TimeoutError))

    def _complete_with_retry(self, prompt: str, system: str, max_tokens: int,
                             temperature: float) -> tuple[str, int, int]:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                return self._complete(prompt, system, max_tokens, temperature)
            except Exception as e:
                if attempt >= self.max_retries or not self._should_retry(e):
                    raise
                self._sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 temperature: float = 0.7, task_id: str = "generic") -> str:
        start = time.monotonic()
        text, in_tok, out_tok = self._complete_with_retry(
            prompt, system, max_tokens, temperature)
        self.logger.log({
            "ts": time.time(),
            "task_id": task_id,
            "model": self.model,
            "prompt_sha256": hashlib.sha256((system + "\n" + prompt).encode()).hexdigest(),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": round(time.monotonic() - start, 3),
        })
        return text

    def complete_json(self, prompt: str, schema: type[BaseModel], system: str = "",
                      max_tokens: int = 2048, temperature: float = 0.7,
                      task_id: str = "generic", parse_retries: int = 3) -> BaseModel:
        """Complete and validate against a pydantic schema, feeding validation
        errors back to the model on retry."""
        attempt_prompt = prompt
        last_err: Exception | None = None
        for _ in range(parse_retries):
            text = self.complete(attempt_prompt, system=system, max_tokens=max_tokens,
                                 temperature=temperature, task_id=task_id)
            try:
                return schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as e:
                last_err = e
                attempt_prompt = (
                    f"{prompt}\n\nYour previous response was invalid: {e}\n"
                    f"Respond again with ONLY a valid JSON object matching the schema."
                )
        raise ValueError(f"could not get valid {schema.__name__} after "
                         f"{parse_retries} attempts: {last_err}")

    def _complete(self, prompt: str, system: str, max_tokens: int,
                  temperature: float) -> tuple[str, int, int]:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, log_path: Path | None = None,
                 max_retries: int = 4):
        super().__init__(model=model, log_path=log_path, max_retries=max_retries)
        import anthropic  # deferred so offline runs never need the key

        self._client = anthropic.Anthropic()
        self._anthropic = anthropic

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, self._anthropic.APIConnectionError):
            return True
        if isinstance(exc, (self._anthropic.APIStatusError,
                            self._anthropic.RateLimitError)):
            return getattr(exc, "status_code", None) in (408, 429, 500, 502, 503, 529)
        return super()._should_retry(exc)

    def _complete(self, prompt: str, system: str, max_tokens: int,
                  temperature: float) -> tuple[str, int, int]:
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class MockLLMClient(LLMClient):
    """Deterministic offline client.

    Parses the ```json block that prompt builders embed (fields: task_id plus
    task inputs) and dispatches to a handler. Handlers are seeded from the
    prompt hash so behavior is reproducible and prompt-sensitive.
    """

    def __init__(self, model: str = "mock", log_path: Path | None = None, seed: int = 0):
        super().__init__(model=model, log_path=log_path)
        self.seed = seed

    def _rng_for(self, prompt: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}:{prompt}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _complete(self, prompt: str, system: str, max_tokens: int,
                  temperature: float) -> tuple[str, int, int]:
        try:
            data = extract_json(prompt)
        except ValueError:
            data = {}
        task = data.get("task_id", "unknown") if isinstance(data, dict) else "unknown"
        handler = getattr(self, f"_handle_{task}", self._handle_unknown)
        out = handler(data, self._rng_for(prompt))
        text = json.dumps(out) if isinstance(out, (dict, list)) else str(out)
        return text, approx_tokens(system + prompt), approx_tokens(text)

    # ----- task handlers ---------------------------------------------------
    def _handle_unknown(self, data: dict, rng: random.Random):
        return {"note": "mock response", "echo_task": data.get("task_id", "unknown")}

    def _handle_generate_session(self, data: dict, rng: random.Random):
        """Write a short dyad conversation session (LLM-mode history gen)."""
        topics = data.get("filler_topics", ["weather", "coffee"])
        turns = []
        for i in range(data.get("n_turns", 8)):
            speaker = "A" if i % 2 == 0 else "B"
            topic = topics[i % len(topics)]
            turns.append({"speaker": speaker,
                          "text": f"I was thinking about the {topic} again today."})
        plant = data.get("plant")
        if plant:
            c = plant["bound_concepts"]
            b = plant["binding_word"]
            listing = ", ".join(f"a {w}" for w in c[:-1]) + f" and a {c[-1]}"
            turns.insert(len(turns) // 2, {
                "speaker": "A",
                "text": f"Strangest thing yesterday: I found a {b} decorated with "
                        f"{listing}, all arranged together on the sidewalk."})
            reply = (f"A {b} with a {c[0]}?! That is amazing, tell me everything "
                     f"about how the {c[-1]} was attached."
                     if plant.get("partner_engaged")
                     else f"Huh, odd. Anyway, about the {topics[0]} thing I mentioned...")
            turns.insert(len(turns) // 2 + 1, {"speaker": "B", "text": reply})
        return {"turns": turns}

    def _handle_rarity_probe(self, data: dict, rng: random.Random):
        """Pick k words 'associated' with the probe word — mock picks
        pseudo-randomly, i.e. at chance, as a no-history model should for a
        genuinely rare binding."""
        words = list(data.get("word_list", []))
        k = data.get("k", 3)
        rng.shuffle(words)
        return {"selected": words[:k]}

    def _handle_summarize_history(self, data: dict, rng: random.Random):
        gists = [f"Session {s['index']}: talked about {', '.join(s['topics'][:2]) or 'various things'}."
                 for s in data.get("session_digests", [])]
        return {"summary": " ".join(gists)}

    def _handle_give_clue(self, data: dict, rng: random.Random):
        team = [w for w in data.get("team_words", [])]
        board = {w.lower() for w in data.get("board_words", [])}
        # prefer a retrieval candidate (graph condition) covering >= 2 targets
        for cand in data.get("candidate_clue_words") or []:
            covered = [t for t in cand.get("covered_targets", []) if t in team]
            if cand["word"].lower() not in board and len(covered) >= 2:
                return {"clue_word": cand["word"], "number": len(covered),
                        "intended_targets": covered,
                        "why_partner_will_get_it": "mock: shared planted episode",
                        "why_eavesdropper_wont": "mock: private association"}
        # otherwise scan memory text for a word that co-occurs with >= 2 team
        # words on one line (finds plants in RAW memory)
        mem = data.get("memory_text", "")
        for line in mem.splitlines():
            tokens = _mock_tokenize(line)
            covered = [t for t in team if t in tokens]
            if len(covered) >= 2:
                from ..engine.wordlist import BINDING_WORD_POOL
                binding = set(BINDING_WORD_POOL)
                clue = next((w for w in tokens if w in binding and w not in board),
                            None) or next((w for w in tokens
                                           if w not in board and len(w) > 3), None)
                if clue:
                    return {"clue_word": clue, "number": min(2, len(covered)),
                            "intended_targets": covered[:2],
                            "why_partner_will_get_it": "mock: co-mentioned in history",
                            "why_eavesdropper_wont": "mock: not a public association"}
        n = min(2, len(team))
        pool = [w for w in ("gondola", "zeppelin", "thimble", "kazoo", "monocle")
                if w not in board]
        return {"clue_word": rng.choice(pool), "number": n,
                "intended_targets": team[:n],
                "why_partner_will_get_it": "mock: generic",
                "why_eavesdropper_wont": "mock: generic"}

    def _handle_guess(self, data: dict, rng: random.Random):
        words = [w for w in data.get("board_words", [])]
        k = data.get("number", 1)
        clue = (data.get("clue_word") or "").lower()
        picks: list[str] = []
        # a partner with memory scans it for lines mentioning the clue word
        # and recovers co-mentioned board words; an eavesdropper has none
        for line in data.get("memory_text", "").splitlines():
            tokens = _mock_tokenize(line)
            if clue and clue in tokens:
                picks.extend(w for w in words if w in tokens and w not in picks)
        rest = [w for w in words if w not in picks]
        rng.shuffle(rest)
        return {"guesses": (picks + rest)[:k], "reasoning": "mock guess"}
