"""
verify_setup.py
================
Phase 1 setup verification script for the AI Travel Planner.

This script validates that:
  1. The Python version meets the minimum requirement (3.11+).
  2. All required third-party packages can be imported.
  3. The project package structure is importable.
  4. Configuration loading works with mock environment variables.
  5. The .env file (if present) does not expose placeholder values.
  6. The .env file is listed in .gitignore.

Run with::

    python verify_setup.py

Exit codes:
    0 - All checks passed.
    1 - One or more checks failed.
"""

from __future__ import annotations

# Force UTF-8 output on Windows to avoid cp1252 encoding errors.
import io
import importlib
import os
import sys
from pathlib import Path
from typing import Callable

# Redirect stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
SKIP = "  [SKIP]"

_results: list[tuple[str, bool | None, str]] = []  # (label, ok, detail)


def check(label: str, detail: str = "", ok: bool = True) -> bool:
    status = PASS if ok else FAIL
    _results.append((label, ok, detail))
    extra = f" – {detail}" if detail else ""
    print(f"{status}  {label}{extra}")
    return ok


def warn(label: str, detail: str = "") -> None:
    extra = f" – {detail}" if detail else ""
    _results.append((label, None, detail))
    print(f"{WARN}  {label}{extra}")


def skip(label: str, detail: str = "") -> None:
    extra = f" – {detail}" if detail else ""
    _results.append((label, None, detail))
    print(f"{SKIP}  {label}{extra}")


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ---------------------------------------------------------------------------
# Check 1: Python Version
# ---------------------------------------------------------------------------


def check_python_version() -> None:
    section("1 · Python Version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    ok = (major, minor) >= (3, 11)
    check(
        f"Python >= 3.11 (found {version_str})",
        ok=ok,
        detail="" if ok else f"Upgrade to Python 3.11+. Current: {version_str}",
    )


# ---------------------------------------------------------------------------
# Check 2: Required Package Imports
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_community", "langchain-community"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("langgraph", "langgraph"),
    ("langsmith", "langsmith"),
    ("langchain_tavily", "langchain-tavily"),
    ("tavily", "tavily-python"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("dotenv", "python-dotenv"),
    ("requests", "requests"),
    ("httpx", "httpx"),
    ("sqlalchemy", "sqlalchemy"),
    ("pytest", "pytest"),
    ("tenacity", "tenacity"),
]


def check_package_imports() -> None:
    section("2 · Package Imports")
    for module_name, pkg_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            check(f"import {module_name}", ok=True)
        except ImportError as exc:
            check(
                f"import {module_name}",
                detail=f"Run: pip install {pkg_name}  ({exc})",
                ok=False,
            )


# ---------------------------------------------------------------------------
# Check 3: Project Package Structure
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

EXPECTED_FILES: list[str] = [
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "README.md",
    "backend/__init__.py",
    "backend/config.py",
    "backend/main.py",
    "backend/agents/__init__.py",
    "backend/graph/__init__.py",
    "backend/tools/__init__.py",
    "backend/models/__init__.py",
    "backend/api/__init__.py",
    "backend/database/__init__.py",
    "frontend/.gitkeep",
    "tests/.gitkeep",
    "docs/.gitkeep",
]


def check_project_structure() -> None:
    section("3 · Project File Structure")
    for rel_path in EXPECTED_FILES:
        path = PROJECT_ROOT / rel_path
        ok = path.exists()
        check(
            str(rel_path),
            ok=ok,
            detail="" if ok else "File not found",
        )


# ---------------------------------------------------------------------------
# Check 4: Backend Package Importability
# ---------------------------------------------------------------------------


def check_backend_imports() -> None:
    section("4 · Backend Package Importability")

    # Ensure the project root is on sys.path.
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    packages = [
        "backend",
        "backend.agents",
        "backend.graph",
        "backend.tools",
        "backend.models",
        "backend.api",
        "backend.database",
    ]
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            check(f"import {pkg}", ok=True)
        except Exception as exc:  # noqa: BLE001
            check(f"import {pkg}", detail=str(exc), ok=False)


# ---------------------------------------------------------------------------
# Check 5: Configuration Loading (with mock env vars)
# ---------------------------------------------------------------------------


def check_config_loading() -> None:
    section("5 · Configuration Loading")

    # Inject mock (but syntactically valid) env vars so the validator passes.
    mock_env = {
        "GOOGLE_API_KEY": "mock-google-key-for-testing-only",
        "TAVILY_API_KEY": "mock-tavily-key-for-testing-only",
        "LANGSMITH_TRACING": "false",
        "APP_ENV": "development",
        "LOG_LEVEL": "DEBUG",
    }

    original_env: dict[str, str | None] = {}
    for key, val in mock_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = val

    try:
        # Clear the lru_cache so Settings is re-constructed with mock values.
        from backend.config import get_settings

        get_settings.cache_clear()

        cfg = get_settings()

        check("Settings object created", ok=True)
        check("google_api_key loaded", ok=bool(cfg.google_api_key))
        check("tavily_api_key loaded", ok=bool(cfg.tavily_api_key))
        check(
            "langsmith_tracing defaults to False",
            ok=cfg.langsmith_tracing is False,
        )
        check("app_env is 'development'", ok=cfg.is_development)
        check(
            "configure_langsmith() runs without error",
            ok=_safe_call(cfg.configure_langsmith),
        )

    except Exception as exc:  # noqa: BLE001
        check("Settings object created", detail=str(exc), ok=False)

    finally:
        # Restore original environment.
        for key, original_val in original_env.items():
            if original_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_val

        # Re-clear so the next import gets clean state.
        try:
            from backend.config import get_settings as _gs

            _gs.cache_clear()
        except Exception:  # noqa: BLE001
            pass


def _safe_call(fn: Callable[[], None]) -> bool:
    try:
        fn()
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Check 6: .env Security
# ---------------------------------------------------------------------------


def check_env_security() -> None:
    section("6 · Environment & Security")

    # 6a – .env must NOT be committed (must be in .gitignore)
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        in_gitignore = ".env" in content
        check(
            ".env is listed in .gitignore",
            ok=in_gitignore,
            detail="" if in_gitignore else "Add '.env' to .gitignore",
        )
    else:
        check(".gitignore exists", ok=False, detail=".gitignore file not found")

    # 6b – If .env exists, make sure it doesn't contain placeholder values.
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        has_placeholder = any(
            placeholder in content
            for placeholder in [
                "your_google_api_key_here",
                "your_tavily_api_key_here",
            ]
        )
        if has_placeholder:
            warn(
                ".env contains placeholder values",
                "Replace placeholder keys with real API keys before use.",
            )
        else:
            check(".env does not contain placeholder values", ok=True)
    else:
        skip(
            ".env file not found (optional for verification)",
            "Copy .env.example to .env and add your API keys.",
        )

    # 6c – Make sure no source file contains a real-looking API key.
    # Heuristic: look for strings that look like Google API keys (AIza…).
    py_files = list((PROJECT_ROOT / "backend").rglob("*.py"))
    found_hardcoded = False
    for f in py_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        if "AIza" in text:
            check(
                f"No hardcoded API key in {f.relative_to(PROJECT_ROOT)}",
                ok=False,
                detail="Remove hardcoded key from source",
            )
            found_hardcoded = True
    if not found_hardcoded:
        check("No hardcoded API keys found in backend source files", ok=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> int:
    section("Summary")
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok is True)
    failed = sum(1 for _, ok, _ in _results if ok is False)
    warned = sum(1 for _, ok, _ in _results if ok is None)

    print(f"\n  Total checks : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")
    print(f"  Warnings     : {warned}")

    if failed == 0:
        print(
            "\n  *** All checks passed! "
            "The project foundation is ready for Phase 2. ***\n"
        )
        return 0
    else:
        print(
            f"\n  !!! {failed} check(s) failed. "
            "Resolve the issues above before proceeding. !!!\n"
        )
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print("\n" + "=" * 60)
    print("  AI Travel Planner - Phase 1 Setup Verification")
    print("=" * 60)

    check_python_version()
    check_package_imports()
    check_project_structure()
    check_backend_imports()
    check_config_loading()
    check_env_security()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
