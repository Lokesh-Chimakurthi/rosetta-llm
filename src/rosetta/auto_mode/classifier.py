"""Ask the configured classifier model for a verdict on each pending tool call.

The classifier is called like any upstream: the request is built as IR and
rendered by the classifier provider's codec, so every provider format works.
Every failure becomes an `Unavailable` verdict, which Claude Code answers by
denying the call; nothing defaults to allowed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import orjson
from pydantic import BaseModel, StrictBool, ValidationError

from rosetta.auto_mode.instructions import (
    AutoModeSetupError,
    Instructions,
    load_instructions,
    render_rules,
)
from rosetta.codecs.registry import PARSE_RESPONSE, RENDER_REQUEST, UPSTREAM_PATH
from rosetta.config import AutoModeConfig, Config, Format
from rosetta.ir.request import CanonicalRequest, Message, RefusalPart, TextPart
from rosetta.observability import get_logger, sub_request_log
from rosetta.routing import resolve_model
from rosetta.upstream import UpstreamClient, UpstreamError, UpstreamResponse

type UnavailableReason = Literal["timeout", "error", "refused", "input_too_long", "truncated"]

# A flagged verdict needs an explanation on the wire; Claude Code shows it as the reason.
_NO_EXPLANATION = (
    "[Unexplained] The auto mode classifier blocked this call without giving a reason."
)


@dataclass(frozen=True, slots=True)
class Evaluated:
    flagged: bool
    explanation: str = ""

    def wire(self) -> dict[str, str]:
        if not self.flagged:
            return {"type": "evaluated", "outcome": "not_flagged"}
        explanation = self.explanation.strip() or _NO_EXPLANATION
        return {"type": "evaluated", "outcome": "flagged", "explanation": explanation}


@dataclass(frozen=True, slots=True)
class Unavailable:
    reason: UnavailableReason

    def wire(self) -> dict[str, str]:
        return {"type": "unavailable", "reason": self.reason}


type Verdict = Evaluated | Unavailable


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: Any


@dataclass(frozen=True, slots=True)
class Session:
    """What Claude Code sent about the session: `classifier_context` and the request's messages."""

    classifier_context: Any
    messages: Sequence[Any]


@dataclass(frozen=True, slots=True)
class ClassifierRoute:
    model_id: str
    provider_key: str
    provider_format: Format
    upstream_model: str


class _ReplyVerdict(BaseModel):
    flagged: StrictBool
    explanation: str = ""


class _Reply(BaseModel):
    verdicts: dict[str, Any]


def prepare(config: Config, auto_mode: AutoModeConfig) -> tuple[Instructions, ClassifierRoute]:
    """Load the instructions and resolve the classifier model; runs at startup.

    Raises:
        AutoModeSetupError: see `load_instructions`, or the model id does not resolve.
    """
    try:
        provider, provider_key, upstream_model = resolve_model(auto_mode.classifier_model, config)
    except ValueError as exc:
        raise AutoModeSetupError(f"auto_mode.classifier_model: {exc}") from exc
    route = ClassifierRoute(
        auto_mode.classifier_model, provider_key, provider.format, upstream_model
    )
    return load_instructions(auto_mode), route


def describe(instructions: Instructions, route: ClassifierRoute) -> dict[str, Any]:
    """What the classifier runs with, without secrets or prompt text; for logs and `check`."""
    builtin = instructions.builtin
    rules: dict[str, Any] | None = None
    if builtin is not None:
        rules = {"claude_code": builtin.claude_version}
        for section, entries in builtin.sections.items():
            rules[section] = len(entries)
    operator_file = instructions.operator_file
    return {
        "classifier_model": route.model_id,
        "builtin_rules": rules,
        "instructions_file": None if operator_file is None else str(operator_file),
    }


class Classifier:
    """Calls the classifier model; safe to share across requests."""

    def __init__(
        self,
        *,
        instructions: Instructions,
        route: ClassifierRoute,
        upstream: UpstreamClient,
        timeout_s: float,
        max_output_tokens: int,
    ) -> None:
        self.instructions = instructions
        self.route = route
        self._upstream = upstream
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens

    async def classify(self, session: Session, calls: Sequence[ToolCall]) -> dict[str, Verdict]:
        """One verdict per call id; never raises for a classifier or provider failure."""
        if not calls:
            return {}
        body = RENDER_REQUEST[self.route.provider_format](self._request(session, calls))
        try:
            async with asyncio.timeout(self._timeout_s):
                with sub_request_log(self.route.upstream_model):
                    resp = await self._upstream.post(
                        self.route.provider_key,
                        UPSTREAM_PATH[self.route.provider_format],
                        orjson.dumps(body),
                        {},
                    )
        except TimeoutError:
            return self._failed(calls, "timeout")
        except UpstreamError as exc:
            return self._failed(calls, "timeout" if exc.kind == "timeout" else "error")
        if not resp.ok:
            return self._failed(calls, _status_reason(resp), upstream_status=resp.status)
        return self._verdicts(resp, calls)

    def _request(self, session: Session, calls: Sequence[ToolCall]) -> CanonicalRequest:
        pending = [{"id": c.id, "name": c.name, "input": c.input} for c in calls]
        context = session.classifier_context
        auto_mode = context.get("auto_mode") if isinstance(context, dict) else None
        text = (
            f"<environment>\n{_json(context)}\n</environment>\n\n"
            f"<rules>\n{render_rules(auto_mode, self.instructions.builtin)}\n</rules>\n\n"
            f"<transcript>\n{transcript(session.messages)}\n</transcript>\n\n"
            f"<tool_calls>\n{_json(pending)}\n</tool_calls>"
        )
        return CanonicalRequest(
            model=self.route.upstream_model,
            system=self.instructions.system_prompt,
            messages=[Message(role="user", parts=[TextPart(text=text)])],
            max_output_tokens=self._max_output_tokens,
        )

    def _verdicts(self, resp: UpstreamResponse, calls: Sequence[ToolCall]) -> dict[str, Verdict]:
        try:
            parsed = PARSE_RESPONSE[self.route.provider_format](resp.json())
        except Exception:  # noqa: BLE001 - an unreadable classifier reply is a verdict-less call, logged
            return self._failed(calls, "error")
        parts = [p for m in parsed.output_messages for p in m.parts]
        if any(isinstance(p, RefusalPart) for p in parts) or parsed.stop.normalized == "refusal":
            return self._failed(calls, "refused")
        reply = _parse_reply("".join(p.text for p in parts if isinstance(p, TextPart)))
        if reply is None:
            return self._failed(calls, "error")
        verdicts = {c.id: _verdict(reply.verdicts.get(c.id)) for c in calls}
        missing = sum(isinstance(v, Unavailable) for v in verdicts.values())
        if missing:
            self._log_failure("error", len(calls), missing=missing)
        return verdicts

    def _failed(
        self,
        calls: Sequence[ToolCall],
        reason: UnavailableReason,
        *,
        upstream_status: int | None = None,
    ) -> dict[str, Verdict]:
        self._log_failure(reason, len(calls), upstream_status=upstream_status)
        return {c.id: Unavailable(reason) for c in calls}

    def _log_failure(
        self,
        reason: UnavailableReason,
        tool_calls: int,
        *,
        missing: int | None = None,
        upstream_status: int | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "reason": reason,
            "classifier_model": self.route.model_id,
            "tool_calls": tool_calls,
        }
        if missing is not None:
            event["missing_verdicts"] = missing
        if upstream_status is not None:
            event["upstream_status"] = upstream_status
        get_logger().warning("auto_mode_review_failed", **event)


def transcript(messages: Sequence[Any]) -> str:
    """The conversation as Claude Code's own classifier sees it, one JSON object per line.

    User text (CLAUDE.md and other context Claude Code attaches to user turns
    included) and the agent's tool calls. Tool results are left out, as
    https://code.claude.com/docs/en/permission-modes#eliminate-prompts-with-auto-mode
    documents for Claude Code's own classifier; so are assistant prose,
    thinking, images and `system` messages, which its classifier requests
    (captured from Claude Code 2.1.319) also omit.
    """
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if role == "user" and block.get("type") == "text":
                lines.append(_json({"user": block.get("text", "")}))
            elif role == "assistant" and block.get("type") == "tool_use":
                lines.append(_json({str(block.get("name", "")): block.get("input")}))
    return "\n".join(lines)


def _json(value: Any) -> str:
    return orjson.dumps(value, default=str).decode()


def _parse_reply(text: str) -> _Reply | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return _Reply.model_validate_json(text[start : end + 1])
    except ValidationError:
        return None


def _verdict(raw: Any) -> Verdict:
    if raw is None:
        return Unavailable("error")
    try:
        verdict = _ReplyVerdict.model_validate(raw)
    except ValidationError:
        return Unavailable("error")
    return Evaluated(verdict.flagged, verdict.explanation)


def _status_reason(resp: UpstreamResponse) -> UnavailableReason:
    if resp.status == 413:
        return "input_too_long"
    try:
        payload = resp.json()
    except ValueError:
        return "error"
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    # OpenAI-compatible providers name an over-long prompt with this code.
    return "input_too_long" if code == "context_length_exceeded" else "error"
