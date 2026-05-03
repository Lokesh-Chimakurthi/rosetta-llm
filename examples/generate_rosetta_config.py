#!/usr/bin/env python3
"""
Rosetta-LLM Config Generator
Generates a config.json for rosetta-llm proxy with a full 3x4 mapping matrix.

Matrix: 3 tiers (opus, sonnet, haiku) × 4 providers (Z.ai, Anthropic, DeepSeek, LM Studio)
Each cell has a default upstream model name. Override provider, model, and client-facing
ID per tier via command-line arguments.

CLAUDE CODE DROP-IN MODE (via Rosetta's alias resolution):
  Each generated tier's model carries `aliases` and an `alias_pattern` so that any of
  Claude Code's bare default model strings — e.g. ANTHROPIC_DEFAULT_OPUS_MODEL set to
  "claude-opus-4-7" — resolve to the matching tier without any client-side change. The
  resolver picks the unique match across providers; ambiguous shorthand returns a 400
  listing every candidate so you know what to disambiguate.

  Defaults installed per tier:
    opus    aliases: ["opus", "claude-opus-4-7", "claude-opus-4-1"]
            alias_pattern: ^claude-opus.*
    sonnet  aliases: ["sonnet", "claude-sonnet-4-5", "claude-sonnet-4-7"]
            alias_pattern: ^claude-sonnet.*
    haiku   aliases: ["haiku", "claude-haiku-4-5"]
            alias_pattern: ^claude-haiku.*

  This means leaving Claude Code's settings completely default and just pointing
  ANTHROPIC_BASE_URL at Rosetta is enough — the bare opus/sonnet/haiku model strings
  it sends will route to whichever upstream you wired up per tier.

CLAUDE CODE MODEL PICKER (still works for explicit selection):
  On startup, Claude Code calls GET /v1/models. Rosetta detects Claude Code (via the
  X-Claude-Code-Session-Id header) and returns a tailored model list:

  - Models whose full ID starts with "claude-" or "anthropic/" pass through unchanged
  - All other models get a "claude-code/" prefix so they pass Claude Code's built-in
    filter (which only shows models starting with "claude" or "anthropic")

  Example: provider key "opus" with model id "claude-opus-4-20250514" → full Rosetta ID
  is "opus/claude-opus-4-20250514", which does NOT start with "claude-", so the picker
  shows it as "claude-code/opus/claude-opus-4-20250514". On selection Rosetta strips
  the "claude-code/" prefix and routes to the "opus" provider's upstream_name.

Usage:
  # Accept all defaults
  python generate_rosetta_config.py

  # Route opus to DeepSeek
  python generate_rosetta_config.py --opus-provider deepseek

  # Override the client-facing model ID
  python generate_rosetta_config.py --haiku-id claude-3-haiku-20240307

  # Full override
  python generate_rosetta_config.py \
    --opus-provider deepseek   --opus-model deepseek-v4-pro \
    --sonnet-provider zai      --sonnet-model GLM-4.7 \
    --haiku-provider deepseek  --haiku-model deepseek-v4-flash

  # Just show the default matrix without generating a file
  python generate_rosetta_config.py --show-matrix
"""

import argparse
import json

# =============================================================================
# UPSTREAM PROVIDER DEFINITIONS
# =============================================================================
PROVIDERS = {
    "zai": {
        "format": "anthropic",
        "base_url": "https://api.z.ai/api/anthropic/v1",
        "api_key_env": "ZAI_API_KEY",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "anthropic": {
        "format": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "deepseek": {
        "format": "openai_chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "lmstudio": {
        "format": "openai_chat",
        "base_url": "http://10.0.0.9:1234/v1",
        # LM Studio's local server doesn't authenticate; placeholder key.
        "api_key": "not-needed",
    },
}

PROVIDER_DISPLAY = {
    "zai":       "Z.ai",
    "anthropic": "Anthropic",
    "deepseek":  "DeepSeek",
    "lmstudio":  "LM Studio",
}

PROVIDER_KEYS = ["zai", "anthropic", "deepseek", "lmstudio"]

# =============================================================================
# 3×4 DEFAULT MAPPING MATRIX
# Rows = tiers (what the client requests), Columns = providers (where it goes)
# Each cell = default upstream model name for that tier+provider combination.
#
# Z.ai model recommendations (researched 2026-05):
#   Opus-tier:   GLM-5.1  — flagship, Opus-4.6-level coding, 8hr autonomy, 200K ctx
#                GLM-5    — 744B MoE (40B active), purpose-built for agentic engineering
#   Sonnet-tier: GLM-4.7  — SOTA balanced, agentic coding, "think before acting", 200K ctx
#                GLM-4.6  — strong coding, versatile, same price as 4.7
#   Haiku-tier:  GLM-4.7-Flash — FREE, 200K ctx, 30B params, high quality for the price
#                GLM-4.7-FlashX — $0.07/$0.40 per 1M tokens, higher throughput
#
# DeepSeek models (as of 2026-05):
#   deepseek-v4-pro   — flagship reasoning model
#   deepseek-v4-flash — fast/cheap, good for haiku-tier workloads
#   deepseek-chat     — legacy, deprecated 2026/07/24
#   deepseek-reasoner — legacy, deprecated 2026/07/24
# =============================================================================
DEFAULT_MATRIX = {
    #            Z.ai              Anthropic                          DeepSeek            LM Studio
    "opus":   {"zai": "GLM-5.1",       "anthropic": "claude-opus-4-20250514",      "deepseek": "deepseek-v4-pro",   "lmstudio": "qwen/qwen3-27b"},
    "sonnet": {"zai": "GLM-4.7",       "anthropic": "claude-sonnet-4-20250514",    "deepseek": "deepseek-v4-pro",   "lmstudio": "qwen/qwen3.6-27b"},
    "haiku":  {"zai": "GLM-4.7-Flash", "anthropic": "claude-haiku-4-5-20251001",   "deepseek": "deepseek-v4-flash", "lmstudio": "qwen/qwen3.6-27b"},
}

# Which provider each tier defaults to
DEFAULT_PROVIDER = {
    "opus":   "zai",
    "sonnet": "anthropic",
    "haiku":  "lmstudio",
}

# The model ID that the client uses to request a tier.
# This is what appears AFTER the provider prefix in the Rosetta model ID format.
# e.g. client sends "opus/claude-opus-4-20250514" and Rosetta matches
# the "claude-opus-4-20250514" part against this id.
DEFAULT_TIER_IDS = {
    "opus":   "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku":  "claude-haiku-4-5-20251001",
}

# Bare-name aliases attached to each tier model. With Rosetta's shorthand
# resolution, any of these resolve to the tier model when the inbound `model`
# field has no provider prefix (which is what Claude Code sends by default).
DEFAULT_TIER_ALIASES = {
    "opus":   ["opus", "claude-opus-4-7", "claude-opus-4-1"],
    "sonnet": ["sonnet", "claude-sonnet-4-5", "claude-sonnet-4-7"],
    "haiku":  ["haiku", "claude-haiku-4-5"],
}

# Regex (Python re, full-string match) attached to each tier so future Anthropic
# date-stamped variants Claude Code might send still resolve to the right tier.
DEFAULT_TIER_ALIAS_PATTERN = {
    "opus":   r"^claude-opus.*",
    "sonnet": r"^claude-sonnet.*",
    "haiku":  r"^claude-haiku.*",
}

TIER_DISPLAY_ORDER = ["opus", "sonnet", "haiku"]


# =============================================================================
# HELPERS
# =============================================================================
def print_matrix(matrix, chosen_provider):
    """Print the full 3x4 matrix, marking which provider is active per tier."""
    col_width = 26

    header = f"{'Tier':<8}"
    for pk in PROVIDER_KEYS:
        header += f"{PROVIDER_DISPLAY[pk]:^{col_width}}"
    print(header)
    print("-" * (8 + col_width * len(PROVIDER_KEYS)))

    for tier in TIER_DISPLAY_ORDER:
        row = f"{tier:<8}"
        for pk in PROVIDER_KEYS:
            model = matrix[tier][pk]
            marker = " <--" if pk == chosen_provider[tier] else ""
            cell = f"{model}{marker}"
            row += f"{cell:^{col_width}}"
        print(row)

    print()


def picker_name(provider_key, tier_id):
    """Compute how a model appears in Claude Code's model picker.

    Rosetta rules:
      - If the full ID starts with "claude-" or "anthropic/" -> shown as-is
      - Otherwise -> gets "claude-code/" prefix
    """
    full_id = f"{provider_key}/{tier_id}"
    if full_id.startswith("claude-") or full_id.startswith("anthropic/"):
        return full_id
    return f"claude-code/{full_id}"


def build_provider_block(provider_key, model_entries):
    """Build a provider block for the Rosetta config carrying every tier
    routed through this provider."""
    base = PROVIDERS[provider_key]
    block = {
        "format": base["format"],
        "base_url": base["base_url"],
    }

    if "api_key" in base:
        block["api_key"] = base["api_key"]
    if "api_key_env" in base:
        block["api_key_env"] = base["api_key_env"]
    if "extra_headers" in base:
        block["extra_headers"] = base["extra_headers"]

    block["models"] = list(model_entries)

    return block


def build_parser():
    """Build the argparse parser with per-tier provider, model, and ID overrides."""
    parser = argparse.ArgumentParser(
        description="Generate a rosetta-llm config.json from a 3x4 tier/provider matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Accept all defaults
  %(prog)s

  # Switch sonnet to DeepSeek
  %(prog)s --sonnet-provider deepseek

  # Override the model ID your client sends (e.g. older Claude version)
  %(prog)s --haiku-id claude-3-haiku-20240307

  # Route everything through DeepSeek
  %(prog)s --opus-provider deepseek   --opus-model deepseek-v4-pro \\
           --sonnet-provider deepseek --sonnet-model deepseek-v4-pro \\
           --haiku-provider deepseek  --haiku-model deepseek-v4-flash

  # Preview the default matrix without writing a file
  %(prog)s --show-matrix

Claude Code model picker:
  Rosetta auto-detects Claude Code and transforms model names for the picker:
  - IDs starting with "claude-" or "anthropic/" pass through unchanged
  - All others get a "claude-code/" prefix (stripped internally on selection)

  Provider keys reflect the actual upstream (zai/anthropic/deepseek/lmstudio).
  With the defaults, picker entries are:
    claude-code/zai/claude-opus-4-20250514         (opus tier -> Z.ai GLM-5.1)
    anthropic/claude-sonnet-4-20250514              (sonnet tier -> Anthropic; no prefix)
    claude-code/lmstudio/claude-haiku-4-5-20251001 (haiku tier -> LM Studio)

  Drop-in mode: each tier model carries `aliases` and `alias_pattern`, so even
  Claude Code's bare default model strings (no provider prefix) resolve to the
  correct tier via Rosetta's shorthand resolver — no client-side override
  required.

available models by provider (as of 2026-05):

  Z.ai:
    Opus-tier:   GLM-5.1 (flagship, 200K), GLM-5 (744B MoE, 200K), GLM-5-Turbo (200K)
    Sonnet-tier: GLM-4.7 (SOTA balanced, 200K), GLM-4.6 (versatile, 200K), GLM-4.5 (128K)
    Haiku-tier:  GLM-4.7-Flash (FREE, 200K), GLM-4.5-Flash (FREE, 200K), GLM-4.7-FlashX

  Anthropic:
    Opus-tier:   claude-opus-4-20250514
    Sonnet-tier: claude-sonnet-4-20250514
    Haiku-tier:  claude-haiku-4-5-20251001

  DeepSeek:
    Opus/Sonnet: deepseek-v4-pro (flagship reasoning)
    Haiku-tier:  deepseek-v4-flash (fast/cheap)
    Legacy:      deepseek-chat, deepseek-reasoner (deprecated 2026/07/24)

  LM Studio:
    Whatever you have loaded locally (e.g. qwen/qwen3-27b, qwen/qwen3.6-27b)
""",
    )

    parser.add_argument(
        "--show-matrix", action="store_true",
        help="Print the default 3x4 matrix and exit without generating a file.",
    )
    parser.add_argument(
        "-o", "--output", default="config.json",
        help="Output filename (default: config.json).",
    )

    for tier in TIER_DISPLAY_ORDER:
        group = parser.add_argument_group(f"{tier} tier")
        group.add_argument(
            f"--{tier}-provider",
            choices=PROVIDER_KEYS,
            default=DEFAULT_PROVIDER[tier],
            help=f"Provider for {tier} (default: {DEFAULT_PROVIDER[tier]}).",
        )
        group.add_argument(
            f"--{tier}-model",
            default=None,
            help=f"Upstream model name sent to the provider (default: depends on provider, see --show-matrix).",
        )
        group.add_argument(
            f"--{tier}-id",
            default=DEFAULT_TIER_IDS[tier],
            help=f"Client-facing model ID for {tier} (default: {DEFAULT_TIER_IDS[tier]}). "
                 f"Client may send '<chosen_provider>/<this_value>' or any of the "
                 f"configured aliases (see --show-matrix).",
        )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---- SHOW-MATRIX MODE ----
    if args.show_matrix:
        print("=" * 112)
        print("  Rosetta-LLM — Default 3x4 Mapping Matrix")
        print("=" * 112)
        print()
        print_matrix(DEFAULT_MATRIX, DEFAULT_PROVIDER)
        print("Client-facing model IDs, picker names, and bare aliases:")
        for tier in TIER_DISPLAY_ORDER:
            provider_key = DEFAULT_PROVIDER[tier]
            rosetta_id = f"{provider_key}/{DEFAULT_TIER_IDS[tier]}"
            picker = picker_name(provider_key, DEFAULT_TIER_IDS[tier])
            aliases = DEFAULT_TIER_ALIASES.get(tier, [])
            pattern = DEFAULT_TIER_ALIAS_PATTERN.get(tier)
            print(f"  {tier:<8} Rosetta ID:   {rosetta_id}")
            print(f"          Picker:       {picker}")
            if aliases:
                print(f"          Bare aliases: {', '.join(aliases)}")
            if pattern:
                print(f"          Alias regex:  {pattern}")
        print()
        print("Z.ai models:")
        print("  Opus:   GLM-5.1  — flagship, Opus-4.6-level coding, 8hr autonomy, $1.40/$4.40 per 1M")
        print("          GLM-5    — 744B MoE (40B active), agentic engineering, $1.00/$3.20")
        print("  Sonnet: GLM-4.7  — SOTA balanced, agentic coding, 200K ctx, $0.60/$2.20")
        print("          GLM-4.6  — strong coding, versatile, $0.60/$2.20")
        print("  Haiku:  GLM-4.7-Flash — FREE, 200K ctx, 30B params")
        print("          GLM-4.7-FlashX — $0.07/$0.40, higher throughput")
        print()
        print("DeepSeek models:")
        print("  Opus/Sonnet: deepseek-v4-pro  — flagship reasoning model")
        print("  Haiku:       deepseek-v4-flash — fast/cheap")
        print("  Legacy:      deepseek-chat, deepseek-reasoner (deprecated 2026/07/24)")
        print()
        return

    # ---- RESOLVE: provider + model + id per tier ----
    chosen = {}
    models = {}
    tier_ids = {}
    for tier in TIER_DISPLAY_ORDER:
        provider = getattr(args, f"{tier}_provider")
        model_override = getattr(args, f"{tier}_model")
        tier_id = getattr(args, f"{tier}_id")

        chosen[tier] = provider
        tier_ids[tier] = tier_id

        if model_override is not None:
            models[tier] = model_override
        else:
            models[tier] = DEFAULT_MATRIX[tier][provider]

    # ---- SHOW FINAL MAPPING ----
    display_matrix = {}
    for tier in TIER_DISPLAY_ORDER:
        display_matrix[tier] = {}
        for pk in PROVIDER_KEYS:
            if pk == chosen[tier]:
                display_matrix[tier][pk] = models[tier]
            else:
                display_matrix[tier][pk] = DEFAULT_MATRIX[tier][pk]

    print()
    print_matrix(display_matrix, chosen)

    # ---- BUILD CONFIG ----
    # Group tiers by chosen provider so each provider gets one block holding
    # every tier-model that routes through it.
    tiers_by_provider = {}
    for tier in TIER_DISPLAY_ORDER:
        tiers_by_provider.setdefault(chosen[tier], []).append(tier)

    config = {
        "host": "0.0.0.0",
        "port": 7860,
        "log_level": "info",
        "providers": {},
    }

    for provider_key, tiers in tiers_by_provider.items():
        model_entries = []
        for tier in tiers:
            entry = {
                "id": tier_ids[tier],
                "upstream_name": models[tier],
            }
            aliases = DEFAULT_TIER_ALIASES.get(tier)
            if aliases:
                entry["aliases"] = aliases
            pattern = DEFAULT_TIER_ALIAS_PATTERN.get(tier)
            if pattern:
                entry["alias_pattern"] = pattern
            model_entries.append(entry)
        config["providers"][provider_key] = build_provider_block(provider_key, model_entries)

    # ---- WRITE ----
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Generated '{args.output}'.")
    print("Copy to ~/.rosetta-llm/config.json (or rename to .jsonc for comments).\n")

    # ---- USAGE INSTRUCTIONS ----
    print("=" * 64)
    print("  HOW TO USE THIS CONFIG WITH CLAUDE CODE")
    print("=" * 64)
    print()
    print("Rosetta model IDs use format: <provider_key>/<model_id>")
    print("Claude Code's model picker will show them as follows:\n")

    for tier in TIER_DISPLAY_ORDER:
        provider_key = chosen[tier]
        upstream = models[tier]
        rosetta_id = f"{provider_key}/{tier_ids[tier]}"
        picker = picker_name(provider_key, tier_ids[tier])
        aliases = DEFAULT_TIER_ALIASES.get(tier, [])
        pattern = DEFAULT_TIER_ALIAS_PATTERN.get(tier)
        print(f"  {tier:<8}")
        print(f"    Rosetta ID:    {rosetta_id}")
        print(f"    Picker:        {picker}")
        print(f"    Routes to:     {PROVIDER_DISPLAY[provider_key]} as '{upstream}'")
        if aliases:
            print(f"    Bare aliases:  {', '.join(aliases)}")
        if pattern:
            print(f"    Alias regex:   {pattern}")
        print()

    print("DROP-IN: Claude Code's bare default model strings (e.g. 'claude-opus-4-7')")
    print("resolve via the alias / alias_pattern fields above. No client-side override")
    print("is needed — just point ANTHROPIC_BASE_URL at this proxy.")
    print()
    print("NOTE: For the picker, IDs that don't start with 'claude-' or 'anthropic/'")
    print("get a 'claude-code/' prefix so they pass Claude Code's built-in filter.")
    print("That prefix is stripped on selection — no extra config needed.")
    print()


if __name__ == "__main__":
    main()
