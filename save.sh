#!/usr/bin/env bash
# ============================================================================
# Exea stage: SAVE  (persist results off-box). The runner snapshots the disk
# separately; this additionally commits results + manifest to git so progress
# is visible on GitHub even if a snapshot is lost. Never hard-fails.
# Requires the fine-grained token the runner injects to have push scope on this
# ONE repo. If no remote/token, it commits locally and skips push.
# ============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [ ! -d .git ]; then
  echo "[save] not a git repo; skipping."
  exit 0
fi

# Only results + run state are committed (data/weights/checkpoints are ignored).
git add -A results state/manifest.json state/last_run_summary.json state/day_counter state/logs 2>/dev/null || true

if git diff --cached --quiet 2>/dev/null; then
  echo "[save] nothing to commit."
else
  STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git -c user.email="exea-runner@localhost" -c user.name="exea-runner" \
      commit -m "results: autonomous run $STAMP" >/dev/null 2>&1 || true
  echo "[save] committed results snapshot."
fi

# Prefer an authenticated push if the runner injected a token, so results
# reach GitHub even when the container has no ambient git credentials. Falls
# back to a plain push (works if creds are pre-configured). Never hard-fails,
# and never prints the token.
PAT="${EXEA_GITHUB_PAT:-${GITHUB_TOKEN:-}}"
USER_NAME="${EXEA_GITHUB_USERNAME:-x-access-token}"
pushed=0
if git remote get-url origin >/dev/null 2>&1; then
  ORIGIN="$(git remote get-url origin)"
  if [ -n "$PAT" ] && printf '%s' "$ORIGIN" | grep -q '^https://github.com/'; then
    REPO_PATH="${ORIGIN#https://github.com/}"
    # Feed credentials via GIT_ASKPASS (token stays in the helper's env, never
    # on any command line — so it can't be read from `ps`/proc cmdline).
    ASKPASS="$(mktemp)"
    printf '#!/bin/sh\ncase "$1" in *[Uu]sername*) printf "%%s" "$EXEA_ASKPASS_USER";; *) printf "%%s" "$EXEA_ASKPASS_PAT";; esac\n' > "$ASKPASS"
    chmod +x "$ASKPASS"
    if EXEA_ASKPASS_USER="$USER_NAME" EXEA_ASKPASS_PAT="$PAT" \
       GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
       git push "https://github.com/${REPO_PATH}" HEAD >/dev/null 2>&1; then
      echo "[save] pushed (token)."; pushed=1
    fi
    rm -f "$ASKPASS"
  fi
  if [ "$pushed" -eq 0 ]; then
    if git push origin HEAD >/dev/null 2>&1; then echo "[save] pushed to origin."; pushed=1; fi
  fi
fi
[ "$pushed" -eq 0 ] && echo "[save] push skipped/failed (no token or offline); results committed locally + on the snapshot."
exit 0
