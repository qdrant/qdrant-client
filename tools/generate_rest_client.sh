#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)/$(dirname "$0")/../"

cd $(mktemp -d)

git clone --sparse --filter=blob:none --depth=1 -b dev git@github.com:qdrant/qdrant.git

cd qdrant
git sparse-checkout add docs/redoc/master

OPENAPI_PATH="$(pwd)/docs/redoc/master/openapi.json"

cd $(mktemp -d)

git clone git@github.com:qdrant/pydantic_openapi_v3.git
cd pydantic_openapi_v3

poetry install

cp $OPENAPI_PATH openapi-qdrant.yaml

PATH_TO_QDRANT_CLIENT=$PROJECT_ROOT

INPUT_YAML=openapi-qdrant.yaml IMPORT_NAME="qdrant_client.http" PACKAGE_NAME=qdrant_openapi_client bash -x scripts/model_data_generator.sh

rm -rf ${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/
mv scripts/output/qdrant_openapi_client ${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/

# extend rest queries
python_models=$(python3 -c 'import json; import qdrant_client.embed as e; print(json.dumps(e.__all__))')

models=($(echo "$python_models" | jq -r '.[]'))

FILE=${PATH_TO_QDRANT_CLIENT}/qdrant_client/http/models/models.py

# Define the import statement to add
IMPORT_STATEMENT="from qdrant_client.embed import *"

# Check if the import statement already exists in the file
if ! grep -q "$IMPORT_STATEMENT" "$FILE"; then
  sed -i "1i$IMPORT_STATEMENT" "$FILE"
fi

isort -w 99 -m 3 --tc --fgw 0 --ca $FILE

# Define the new types to add
NEW_TYPES=""
for type in "${models[@]}"; do
  NEW_TYPES+="    ${type},\n"
done

NEW_TYPES_DICT=""
for type in "${models[@]}"; do
  NEW_TYPES_DICT+="    Dict[StrictStr, ${type}],\n"
done

NEW_TYPES_BATCH=""
for type in "${models[@]}"; do
  NEW_TYPES_BATCH+="    List[${type}],\n    Dict[StrictStr, List[${type}]],\n"
done

# Use sed to insert the new types before the last bracket
sed -i '/^VectorStruct = Union\[/,/^]/ s/\(^]\)/'"$NEW_TYPES"'\1/' $FILE
sed -i '/^VectorStruct = Union\[/,/^]/ s/\(^]\)/'"$NEW_TYPES_DICT"'\1/' $FILE
sed -i '/^VectorInput = Union\[/,/^]/ s/\(^]\)/'"$NEW_TYPES"'\1/' $FILE
sed -i '/^BatchVectorStruct = Union\[/,/^]/ s/\(^]\)/'"${NEW_TYPES_BATCH}"'\1/' $FILE
# end extend rest queries
