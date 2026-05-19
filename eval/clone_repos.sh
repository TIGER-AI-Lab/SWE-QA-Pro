#!/bin/bash
# Clone every repository referenced by the SWE-QA-Pro Bench at its evaluation
# commit. Each line of repos.txt is "<git_url> <commit_hash>".
#
# By default repos are placed under ./repos/, matching the --repo-root default
# used by scripts/run_agent.py and scripts/run_direct.py. Override with the
# REPO_FILE or TARGET_DIR env vars if you keep them elsewhere.
set -euo pipefail

REPO_FILE="${REPO_FILE:-./repos.txt}"
TARGET_DIR="${TARGET_DIR:-./repos}"

if [[ ! -f "$REPO_FILE" ]]; then
    echo "ERROR: repo list not found: $REPO_FILE" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

while read -r repo_url commit_hash; do
    # Skip blank lines and comments
    [[ -z "${repo_url:-}" || "${repo_url:0:1}" == "#" ]] && continue

    repo_name=$(basename "$repo_url" .git)
    repo_path="$TARGET_DIR/$repo_name"

    if [[ -d "$repo_path/.git" ]]; then
        echo "[clone_repos] $repo_name already exists, fetching updates..."
        git -C "$repo_path" fetch --quiet origin
    else
        echo "[clone_repos] cloning $repo_name ..."
        git clone --quiet "$repo_url" "$repo_path"
    fi

    git -C "$repo_path" checkout --quiet "$commit_hash"
done < "$REPO_FILE"

echo "[clone_repos] done. All repositories cloned into $TARGET_DIR and checked out at the benchmark commits."
