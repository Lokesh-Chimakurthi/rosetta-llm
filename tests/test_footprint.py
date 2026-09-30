"""Import-time memory budget for the proxy.

Measured in fresh interpreters so the result is independent of test order.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="ru_maxrss is POSIX-only")

# Measured 2026-09-30 on Darwin arm64, CPython 3.12: import delta 34.1 MiB.
# Budget = measured + 10%. Later plans may only lower it.
IMPORT_RSS_BUDGET_MIB = 37.5

_HEAVY_MODULES = ("rich", "tiktoken")


def _peak_rss_mib(code: str) -> float:
    script = f"{code}\nimport resource\nprint(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"
    samples = []
    for _ in range(3):
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", script], check=True, capture_output=True, text=True
        ).stdout
        samples.append(int(out.strip().splitlines()[-1]))
    # macOS reports ru_maxrss in bytes, Linux in KiB.
    unit = 1 if sys.platform == "darwin" else 1024
    return min(samples) * unit / (1024 * 1024)


def test_import_rss_within_budget() -> None:
    delta = _peak_rss_mib("import rosetta.app") - _peak_rss_mib("pass")
    assert delta <= IMPORT_RSS_BUDGET_MIB, (
        f"import rosetta.app grew peak RSS by {delta:.1f} MiB (budget {IMPORT_RSS_BUDGET_MIB})"
    )


def test_import_leaves_heavy_modules_unloaded() -> None:
    code = (
        "import sys, rosetta.app\n"
        f"print(','.join(sorted(set({_HEAVY_MODULES!r}) & set(sys.modules))))"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    assert out.strip() == ""
