"""Rosetta's answer to Claude Code's auto mode review.

In auto mode Claude Code sends `safeguards` with each request and denies any
tool call that comes back without a verdict in `safeguard_results`. Providers
that cannot review get their verdicts from a classifier model named in
`auto_mode.classifier_model`: `instructions` builds what it is told,
`classifier` asks it, `review` attaches the verdicts to the response.
"""
