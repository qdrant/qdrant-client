#!/usr/bin/env sh

set -xe

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

pip install qdrant-client

pip install sphinx==4.5.0
pip install "git+https://github.com/qdrant/qdrant_sphinx_theme.git@master#egg=qdrant-sphinx-theme"
pip install nbsphinx==0.9.3

sphinx-build docs/source docs/html