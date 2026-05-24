#!/usr/bin/env bash
# Poll until portfolio training finishes, then alert (macOS notification + sound).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/train_portfolio.log"

echo "Watching $LOG (Ctrl+C to stop)..."
while true; do
  if grep -q "Both models trained" "$LOG" 2>/dev/null; then
    MSG="Portfolio training finished. Run: bash scripts/post_train_pipeline.sh"
    echo "$MSG"
    if command -v osascript >/dev/null; then
      osascript -e "display notification \"$MSG\" with title \"FastText training done\" sound name \"Glass\""
    fi
    exit 0
  fi
  if ! pgrep -f "train_portfolio_models.py|train.py --epochs 10 --embedding-dim 300" >/dev/null 2>&1; then
    if grep -q "Both models trained" "$LOG" 2>/dev/null; then
      exit 0
    fi
    echo "Training process stopped but log does not show success — check $LOG"
    exit 1
  fi
  sleep 120
done
