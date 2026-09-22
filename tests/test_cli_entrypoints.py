"""Every README command must actually run as `python -m <module>`.

The stage tests call `main(argv)` directly, which passes even when the module has no
`if __name__ == "__main__"` block. `extract.run` shipped without one: `python -m extract.run`
imported the module, ran nothing, printed nothing and exited 0 -- a silent no-op that looked
like success. These tests invoke the modules the way the README tells a user to.
"""

import subprocess
import sys

import pytest

# The four invocations the README's Run section promises.
CLI_MODULES = ("data.prepare", "extract.run", "confidence.fit", "eval.report", "eval.errors")


@pytest.mark.parametrize("module", CLI_MODULES)
def test_module_runs_as_a_script_and_prints_usage(module: str) -> None:
    """`python -m <module> --help` must reach argparse, not import and fall off the end."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"{module} --help exited {result.returncode}: {result.stderr}"
    assert "usage:" in result.stdout, (
        f"{module} produced no argparse usage text; it probably has no "
        f'`if __name__ == "__main__"` block and did nothing at all'
    )


@pytest.mark.parametrize("module", CLI_MODULES)
def test_module_rejects_an_unknown_flag(module: str) -> None:
    """An unrecognised flag must fail loudly, proving argument parsing really ran."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--definitely-not-a-flag"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, f"{module} accepted an unknown flag instead of failing"
