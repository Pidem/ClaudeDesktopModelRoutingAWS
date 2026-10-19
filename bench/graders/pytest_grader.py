"""Code grader (T08): drop the model's code as solution.py next to the hidden
pytest suite plus any fixtures, run pytest, pass iff every test passes."""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

from .base import GradeContext, GradeResult, extract_code


def grade(task: dict, response_text: str, ctx: GradeContext) -> GradeResult:
    args = task.get("grader_args", {})
    code = extract_code(response_text, "python")
    if not code or "def " not in code:
        return GradeResult(False, 0.0, "no python code found in response")

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "solution.py").write_text(code)
        # pytest only collects files named test_*.py.
        shutil.copy(ctx.gold_dir / args["test"], d / "test_de.py")
        for fx in args.get("fixtures", []):
            shutil.copy(ctx.fixtures_dir / fx, d / fx)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_de.py", "-q", "-p", "no:cacheprovider"],
            cwd=d, capture_output=True, text=True, timeout=120,
        )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    passed = proc.returncode == 0
    return GradeResult(passed, 1.0 if passed else 0.0, f"pytest: {tail}")
