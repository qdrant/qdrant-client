#!/usr/bin/env sh

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"


# From: https://github.com/edemaine/topp/blob/main/netlify-pandoc.sh, 3 years ago
PANDOC_URL="https://github.com/jgm/pandoc/releases/download/2.11/pandoc-2.11-1-amd64.deb"

set -o errexit

# We use $DEPLOY_URL to detect the Netlify environment.
if [ -v DEPLOY_URL ]; then
  : ${NETLIFY_BUILD_BASE="/opt/buildhome"}
else
  : ${NETLIFY_BUILD_BASE="$PWD/buildhome"}
fi

NETLIFY_CACHE_DIR="$NETLIFY_BUILD_BASE/cache"
PANDOC_DIR="$NETLIFY_CACHE_DIR/pandoc"
PANDOC_DEB="`basename "$PANDOC_URL"`"
PANDOC_SUCCESS="$NETLIFY_CACHE_DIR/${PANDOC_DEB}-success"
PANDOC_BIN="$PANDOC_DIR/usr/bin"

if [ ! -e "$PANDOC_SUCCESS" ]; then
  curl -L "$PANDOC_URL" -o "$PANDOC_DEB"
  dpkg -x "$PANDOC_DEB" "$PANDOC_DIR"
  touch "$PANDOC_SUCCESS"
fi

exec "$PANDOC_BIN/pandoc" "$@"

# sudo apt-get install -y pandoc
poetry run sphinx-apidoc --force --separate --no-toc -o docs/source qdrant_client
poetry run sphinx-build docs/source docs/html