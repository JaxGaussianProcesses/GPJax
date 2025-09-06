#!/usr/bin/env python3
"""Format GPJax codebase using black, isort, and ruff."""

import subprocess
import sys


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ {description} failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"✅ {description} completed!")


def main() -> None:
    """Run all formatting commands."""
    commands = [
        (["black", "./gpjax", "./tests"], "Running black"),
        (
            ["jupytext", "--pipe", "black", "examples/*.py"],
            "Running jupytext on examples",
        ),
        (["isort", "./gpjax", "./tests"], "Running isort on main code"),
        (
            [
                "isort",
                "examples/*.py",
                "--treat-comment-as-code",
                "# %%",
                "--float-to-top",
            ],
            "Running isort on examples",
        ),
        (["ruff", "format", "./gpjax", "./tests", "./examples"], "Running ruff format"),
    ]

    for cmd, description in commands:
        run_command(cmd, description)

    print("🎉 All formatting complete!")


if __name__ == "__main__":
    main()
