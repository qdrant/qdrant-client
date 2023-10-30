#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

brew install pandoc

pip install qdrant-client

pip install sphinx==4.5.0
pip install ipython
pip install "git+https://github.com/qdrant/qdrant_sphinx_theme.git@master#egg=qdrant-sphinx-theme"
pip install nbsphinx==0.9.3
pip install --upgrade Pygments==2.16.1
sphinx-build docs/source docs/html