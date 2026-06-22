from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Literal

FORBIDDEN_IMPORTS = frozenset({"requests", "urllib", "socket", "subprocess", "http"})

ErrorCode = Literal[
    "SYNTAX_ERROR",
    "FORBIDDEN_IMPORT",
    "RUNTIME_ERROR",
    "TIMEOUT",
    "OUTPUT_VALIDATION_FAILED",
]


@dataclass
class SandboxResult:
    success: bool
    phase: Literal["syntax", "execute", "output"]
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    error: str | None
    error_code: ErrorCode | None


def _check_forbidden_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    violations.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in FORBIDDEN_IMPORTS:
                violations.append(f"from {node.module} import ... (line {node.lineno})")
    return violations


def _minimal_env() -> dict[str, str]:
    return {"PATH": os.environ.get("PATH", "")}


def _wrap_code(code: str, sample_data_path: str) -> str:
    preamble = textwrap.dedent(f"""\
        SAMPLE_DATA_PATH = r"{sample_data_path}"
    """)
    return preamble + code


class SandboxRunner:
    def run(
        self,
        code: str,
        sample_data_path: str,
        timeout_s: int = 30,
    ) -> SandboxResult:
        try:
            ast.parse(code)
        except SyntaxError as e:
            return SandboxResult(
                success=False,
                phase="syntax",
                stdout="",
                stderr="",
                exit_code=None,
                timed_out=False,
                error=f"SyntaxError: {e}",
                error_code="SYNTAX_ERROR",
            )

        violations = _check_forbidden_imports(code)
        if violations:
            return SandboxResult(
                success=False,
                phase="syntax",
                stdout="",
                stderr="",
                exit_code=None,
                timed_out=False,
                error=f"ForbiddenImport: {', '.join(violations)}",
                error_code="FORBIDDEN_IMPORT",
            )

        wrapped = _wrap_code(code, sample_data_path)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=_minimal_env(),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                phase="execute",
                stdout="",
                stderr="",
                exit_code=None,
                timed_out=True,
                error=f"TimeoutExpired: exceeded {timeout_s}s",
                error_code="TIMEOUT",
            )

        if proc.returncode != 0:
            return SandboxResult(
                success=False,
                phase="execute",
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
                error=proc.stderr.strip() or f"Process exited with code {proc.returncode}",
                error_code="RUNTIME_ERROR",
            )

        if not proc.stdout.strip():
            return SandboxResult(
                success=False,
                phase="output",
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                timed_out=False,
                error="Output validation failed: stdout is empty",
                error_code="OUTPUT_VALIDATION_FAILED",
            )

        return SandboxResult(
            success=True,
            phase="output",
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            timed_out=False,
            error=None,
            error_code=None,
        )
