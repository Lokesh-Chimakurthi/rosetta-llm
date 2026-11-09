## ADR-0001: Rosetta answers Claude Code's auto mode review with a configured classifier

Status: Accepted
Date: 2026-11-09

Context: In auto mode Claude Code sends `safeguards` with each Messages request and expects a verdict per tool call in `safeguard_results`. Only the Anthropic API (and providers that forward to it) produce verdicts. Measured against Claude Code 2.1.319 through Rosetta with an `openai_chat` provider:

- A response with no `safeguard_results` gets the call denied ("safety check isn't available").
- A `{"type": "unsupported"}` status gets it denied ("no local one judges in its place").
- A `400` naming `safeguards` also gets it denied. That `400` is Anthropic's documented refusal for gateways that cannot review. After it, Claude Code stops asking for the session, but it uses its own classifier only when started with `CLAUDE_CODE_AUTO_MODE_SERVER=0`.
- An `available` status with per-tool `evaluated` verdicts is obeyed. A `flagged` explanation is shown to the user as the reason.

So auto mode is unusable on any translated route unless Rosetta produces the verdicts.

Claude Code sends no classifier prompt with a review request: in server mode Anthropic's server owns it. What it does send is `classifier_context`: permission rules, the user's `autoMode` lists, trusted directories and git state. In those lists `"$defaults"` arrives literally, and an empty list means "use the built-ins". Its built-in rules are printed by `claude auto-mode defaults`. No CLI command prints its classifier prompt. Its own classifier requests (with `CLAUDE_CODE_AUTO_MODE_SERVER=0`) carry user text, tool calls and CLAUDE.md, never tool results or assistant prose.

Considered options:
- Drop `safeguards` and send no verdicts (the previous behaviour). Rejected: it worked only while Claude Code fell back to its own classifier, which 2.1.319 ended.
- Answer `400` naming `safeguards` and let Claude Code classify. Rejected as the only behaviour: it needs `CLAUDE_CODE_AUTO_MODE_SERVER=0` on every client, a setting Anthropic calls temporary, and the classifier model would then be chosen client-side. It is kept as the answer when `auto_mode` is not configured.
- Copy Claude Code's 142k-character classifier prompt into Rosetta. Rejected: the text belongs to Anthropic, costs about 32k tokens per check, and would go stale silently on every Claude Code release.
- Classify with a configured model, as LiteLLM (`safeguards_classifier_model`) and AxonHub do.

Decision: When `auto_mode.classifier_model` is set, Rosetta answers the review for every route except an `anthropic` provider left at `auto_mode_review: "upstream"`. It strips `safeguards` and its beta before forwarding, then asks the classifier model once per response that contains tool calls, and adds `safeguard_results` to the response or to the stream's final `message_delta`. The classifier's system prompt has three parts, in this order:

1. Rosetta's short base prompt: role, how each rule section applies, reply format.
2. Claude Code's built-in rules, loaded at startup with `claude auto-mode defaults`.
3. The operator's optional `instructions_file`.

The user message carries `classifier_context` verbatim, the per-section rule merge Claude Code applies, and a transcript composed like Claude Code's own: user text and tool calls. Nothing is trimmed. Every failure is an `unavailable` verdict. Without `auto_mode`, such a request gets the documented `400`.

Consequences:
- Auto mode works on every provider format, with the policy (rules) coming from Claude Code and the operator. Rosetta owns only the base prompt.
- Rosetta is now a security control. A wrong `not_flagged` lets a dangerous call run, so every classifier failure is mapped to `unavailable`, which Claude Code denies.
- Each tool-call turn costs one extra model call and waits for it; streams send pings meanwhile.
- The conversation and `classifier_context` go to the classifier's provider.
- Startup depends on the Claude Code CLI unless `builtin_rules` is `none`. Rules follow the CLI installed on the server, which can differ from the client's version.
- The `safeguard_results` shape is undocumented by Anthropic. If it drifts, Claude Code denies every call (fail closed), and the tests pin the shape that 2.1.319 accepted.
