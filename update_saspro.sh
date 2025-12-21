#!/bin/bash
set -e

# Check if branch argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <branch>"
    echo "Example: $0 main"
    echo "         $0 dev"
    exit 1
fi

# Get branch name from argument
BRANCH="$1"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine which pip to use (prefer venv pip if available)
if [ -n "$VIRTUAL_ENV" ]; then
    # Use pip from already activated venv
    PIP_CMD="$VIRTUAL_ENV/bin/pip"
elif [ -d "venv" ]; then
    # Use pip from venv directory
    PIP_CMD="venv/bin/pip"
elif [ -d ".venv" ]; then
    # Use pip from .venv directory
    PIP_CMD=".venv/bin/pip"
elif [ -d "env" ]; then
    # Use pip from env directory
    PIP_CMD="env/bin/pip"
else
    # Fall back to python -m pip
    PIP_CMD="python -m pip"
fi

echo "Fetching latest changes from origin..."
git fetch origin

echo "Switching to branch '$BRANCH'..."
git switch "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

echo "Pulling latest changes (fast-forward only)..."
git pull --ff-only origin "$BRANCH"

echo ""
echo "Installing/updating package..."
$PIP_CMD install --force-reinstall .

echo ""
echo "✅ Updated to latest on '$BRANCH' and reinstalled."

