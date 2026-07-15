#!/usr/bin/env bash
# Installs the "transcribe" skill into Codex and/or Claude Code.
# Usage:  curl -fsSL https://raw.githubusercontent.com/ArLeyar/transcribe-skill/main/install.sh | bash
#    or:  ./install.sh          (from a checkout of this repo)
set -euo pipefail

REPO_URL="https://github.com/ArLeyar/transcribe-skill.git"
say() { printf '\033[1;32m==>\033[0m %s\n' "$1"; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$1" >&2; exit 1; }

# --- source: local checkout if SKILL.md is next to this script, else clone ---
SRC=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/SKILL.md" ]; then
  SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  command -v git >/dev/null || die "git is required. Install the Xcode Command Line Tools: xcode-select --install"
  SRC="$(mktemp -d)/transcribe-skill"
  say "Downloading the skill..."
  git clone --depth 1 --quiet "$REPO_URL" "$SRC"
fi

# --- platform ---
[ "$(uname -s)" = "Darwin" ] || die "macOS is required (the local engine only runs on Apple Silicon)."
if [ "$(uname -m)" != "arm64" ]; then
  echo "WARNING: not Apple Silicon. The local engine will not run; only -e openai will work."
fi

# --- ffmpeg (mandatory, all engines) ---
if command -v ffmpeg >/dev/null; then
  say "ffmpeg already installed"
else
  command -v brew >/dev/null || die "Homebrew not found. Install it from https://brew.sh, then run this installer again."
  say "Installing ffmpeg (takes a couple of minutes)..."
  brew install ffmpeg
fi

# --- uv (runs the script + its python deps) ---
if command -v uv >/dev/null; then
  say "uv already installed"
else
  say "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv installed but is not on PATH. Close and reopen your terminal, then run this again."

# --- targets: every agent host that exists; default to Codex ---
TARGETS=()
[ -d "$HOME/.codex" ] && TARGETS+=("$HOME/.codex/skills/transcribe")
[ -d "$HOME/.claude" ] && TARGETS+=("$HOME/.claude/skills/transcribe")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("$HOME/.codex/skills/transcribe")

for DEST in "${TARGETS[@]}"; do
  # Never blow away a skills dir that some git repo manages (dotfiles setups symlink it).
  if git -C "$(dirname "$DEST")" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "SKIPPED: $DEST lives inside a git repo, leaving it alone. Install manually if you need to."
    continue
  fi
  rm -rf "$DEST"
  mkdir -p "$DEST/scripts"
  cp "$SRC"/scripts/*.py "$DEST/scripts/"
  # SKILL.md ships with a placeholder because the install path differs per host
  sed "s|__SKILL_DIR__|$DEST|g" "$SRC/SKILL.md" > "$DEST/SKILL.md"
  say "Installed: $DEST"
done

cat <<'EOF'

Done. Restart Codex (or Claude Code) and tell it:

    transcribe /path/to/file.m4a

The first run downloads the model (~1.6GB). One time only, then it works offline.
Drag an audio file into the chat window to get its path.
EOF
