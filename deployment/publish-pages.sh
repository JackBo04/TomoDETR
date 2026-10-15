#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Commit your changes before publishing.' >&2
  exit 1
fi
# Publishes only the static site to the configured GitHub Pages branch.
git subtree push --prefix site origin gh-pages
