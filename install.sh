#!/usr/bin/env bash
# Installs the "transcribe" skill into Codex and/or Claude Code.
# Usage:  curl -fsSL https://raw.githubusercontent.com/ArLeyar/transcribe-skill/main/install.sh | bash
#    or:  ./install.sh          (from a checkout of this repo)
set -euo pipefail

REPO_URL="https://github.com/ArLeyar/transcribe-skill.git"
say() { printf '\033[1;32m==>\033[0m %s\n' "$1"; }
die() { printf '\033[1;31mОшибка:\033[0m %s\n' "$1" >&2; exit 1; }

# --- source: local checkout if SKILL.md is next to this script, else clone ---
SRC=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/SKILL.md" ]; then
  SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  command -v git >/dev/null || die "нужен git. Поставь Xcode Command Line Tools: xcode-select --install"
  SRC="$(mktemp -d)/transcribe-skill"
  say "Скачиваю скилл..."
  git clone --depth 1 --quiet "$REPO_URL" "$SRC"
fi

# --- platform ---
[ "$(uname -s)" = "Darwin" ] || die "нужен macOS (локальный движок работает только на Apple Silicon)."
if [ "$(uname -m)" != "arm64" ]; then
  echo "ВНИМАНИЕ: это не Apple Silicon. Локальный движок не заведётся, останется только -e openai."
fi

# --- ffmpeg (mandatory, all engines) ---
if command -v ffmpeg >/dev/null; then
  say "ffmpeg уже есть"
else
  command -v brew >/dev/null || die "нет Homebrew. Открой https://brew.sh, поставь его, потом запусти установку снова."
  say "Ставлю ffmpeg (пару минут)..."
  brew install ffmpeg
fi

# --- uv (runs the script + its python deps) ---
if command -v uv >/dev/null; then
  say "uv уже есть"
else
  say "Ставлю uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv поставился, но не виден. Закрой и открой терминал, запусти установку снова."

# --- targets: every agent host that exists; default to Codex ---
TARGETS=()
[ -d "$HOME/.codex" ] && TARGETS+=("$HOME/.codex/skills/transcribe")
[ -d "$HOME/.claude" ] && TARGETS+=("$HOME/.claude/skills/transcribe")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("$HOME/.codex/skills/transcribe")

for DEST in "${TARGETS[@]}"; do
  # Never blow away a skills dir that some git repo manages (dotfiles setups symlink it).
  if git -C "$(dirname "$DEST")" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "ПРОПУСК: $DEST лежит внутри git-репозитория — не трогаю. Поставь вручную, если надо."
    continue
  fi
  rm -rf "$DEST"
  mkdir -p "$DEST/scripts"
  cp "$SRC"/scripts/*.py "$DEST/scripts/"
  # SKILL.md ships with a placeholder because the install path differs per host
  sed "s|__SKILL_DIR__|$DEST|g" "$SRC/SKILL.md" > "$DEST/SKILL.md"
  say "Установлено: $DEST"
done

cat <<'EOF'

Готово. Перезапусти Codex (или Claude Code) и скажи ему:

    расшифруй /путь/к/файлу.m4a

Первый запуск качает модель (~1.6 ГБ) — это один раз, дальше работает офлайн.
Просто перетащи аудиофайл в окно чата, чтобы получить его путь.
EOF
