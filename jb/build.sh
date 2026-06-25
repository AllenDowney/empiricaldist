#!/bin/bash
# Build Jupyter Book + MkDocs API reference. Run from repo root.
# Pass --no-push to skip ghp-import (local preview only).

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/jb"

cp ../empiricaldist/*.ipynb .
jb build .

cd "$ROOT"
mkdocs build
cp -r site/ jb/_build/html/docs

if [ "${1:-}" != "--no-push" ]; then
  ghp-import -n -p -f jb/_build/html
fi
