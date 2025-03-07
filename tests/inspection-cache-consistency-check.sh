set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

# Keep current version of file to check
ROOT_DIR=$(pwd)
CLIENT_DIR=$ROOT_DIR/qdrant_client
cd "$CLIENT_DIR"

inspection_cache_file=embed/_inspection_cache.py

if [ ! -f $inspection_cache_file ]; then
    echo "ERROR: $inspection_cache_file not found."
    exit 1
fi

cp $inspection_cache_file $inspection_cache_file.diff

"$ROOT_DIR"/tools/populate_inspection_cache.sh

# Ensure generated files are the same as files in this repository

if diff -wa $inspection_cache_file{,.diff}
  then
    set +x
    echo "No diffs found."
  else
    set +x
    echo "ERROR: Generated $file is not consistent with file in this repository, see diff above."
    exit 1
fi

# Cleanup
rm $inspection_cache_file.diff
