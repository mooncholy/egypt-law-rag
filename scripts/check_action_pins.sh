#!/usr/bin/env bash
# Check that every `uses: owner/repo@ref` in the given workflow files names a
# ref that exists on GitHub, so a bad pin fails the commit instead of CI.
#
# - A tag or branch pin must exist in the action's repository.
# - A SHA pin must be the commit of a released tag, and a trailing
#   "# vX.Y.Z" comment must name that same commit.
# - Local actions (./...) and docker:// images are skipped.
#
# Usage: scripts/check_action_pins.sh .github/workflows/*.yml
# Needs network access; offline, skip it with SKIP=check-action-pins.
set -euo pipefail
# A missing repo makes GitHub ask for credentials; fail instead of hanging.
export GIT_TERMINAL_PROMPT=0

declare -A listing_of # repo -> its `git ls-remote` output
declare -A seen       # file + pinned line, so each problem is reported once
declare -A dead_repo  # repos already reported unreachable
status=0

fail() {
  echo "$1"
  status=1
}

for file in "$@"; do
  while IFS= read -r line; do
    spec=$(grep -oP 'uses:\s*\K[^\s#]+' <<<"$line") || continue
    [[ $spec == ./* || $spec == docker://* ]] && continue
    key="$file|$(grep -oP 'uses:.*' <<<"$line")"
    [[ -n ${seen[$key]+set} ]] && continue
    seen[$key]=1
    action=${spec%@*}
    ref=${spec##*@}
    repo=$(cut -d/ -f1,2 <<<"$action") # owner/repo/sub/dir -> owner/repo

    [[ -n ${dead_repo[$repo]+set} ]] && continue
    if [[ -z ${listing_of[$repo]+set} ]]; then
      if ! listing_of[$repo]=$(git ls-remote "https://github.com/$repo" 2>&1); then
        if grep -qiE 'not found|could not read Username' <<<"${listing_of[$repo]}"; then
          fail "$file: $spec: no public repository github.com/$repo"
        else
          fail "$file: $spec: can't reach github.com (offline? SKIP=check-action-pins)"
        fi
        dead_repo[$repo]=1 # report the repo once, not per line
        continue
      fi
    fi
    listing=${listing_of[$repo]}

    if [[ $ref =~ ^[0-9a-f]{40}$ ]]; then
      if ! grep -qP "^$ref\trefs/tags/" <<<"$listing"; then
        fail "$file: $spec: SHA is not the commit of any tag in $repo"
        continue
      fi
      comment=$(grep -oP '#\s*\K\S+' <<<"$line") || continue
      if ! grep -qP "^$ref\trefs/tags/\Q$comment\E(\^\{\})?$" <<<"$listing"; then
        fail "$file: $spec: comment says $comment, but $comment is a different commit"
      fi
    elif ! grep -qP "\trefs/(tags|heads)/\Q$ref\E$" <<<"$listing"; then
      newest=$(git ls-remote --tags --sort=-version:refname "https://github.com/$repo" |
        grep -oP 'refs/tags/\K[^^]+$' | head -1)
      fail "$file: $spec: no tag or branch '$ref' in $repo (newest tag: $newest)"
    fi
  done < <(grep -E '^\s*(-\s*)?uses:' "$file")
done

exit "$status"
