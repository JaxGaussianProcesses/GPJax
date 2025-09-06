#!/bin/bash
# Format GPJax codebase
set -e

echo "🔧 Running black..."
uv run black ./gpjax ./tests

echo "📝 Running jupytext on examples..."
uv run jupytext --pipe black examples/*.py

echo "📦 Running isort on main code..."
uv run isort ./gpjax ./tests

echo "📦 Running isort on examples..."
uv run isort examples/*.py --treat-comment-as-code '# %%' --float-to-top

echo "✨ Running ruff format..."
uv run ruff format ./gpjax ./tests ./examples

echo "✅ Formatting complete!"
