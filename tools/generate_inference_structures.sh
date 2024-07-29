#!/bin/bash

OPENAPI_PATH=$1

validate_json() {
  echo "$1" | jq empty
}

DOCUMENT_TYPE='{
  "type": "object",
  "required": [
    "text"
  ],
  "properties": {
    "text": {
      "type": "string",
      "description": "Text document to be embedded by FastEmbed or Cloud inference server"
    }
  }
}'

DOCUMENT_FLAT_EXTENSION='[{
  "$ref": "#/components/schemas/Document"
}]'

DOCUMENT_ARRAY_EXTENSION='[{
  "type": "array",
  "items": {
    "$ref": "#/components/schemas/Document"
  }
}]'

OPENAPI_SOURCE=$(jq '.' $OPENAPI_PATH)

validate_json "$OPENAPI_SOURCE"
if [ $? -ne 0 ]; then
  echo "Invalid JSON: $OPENAPI_SOURCE"
  exit 1
fi

SCHEMAS=$(echo $OPENAPI_SOURCE | jq -r ".components.schemas")
VECTOR_ANY_OF=$(echo $SCHEMAS | jq -r ".Vector.anyOf")
VECTOR_STRUCT_ANY_OF=$(echo $SCHEMAS | jq -r ".VectorStruct.anyOf")
BATCH_VECTOR_STRUCT_ANY_OF=$(echo $SCHEMAS | jq -r ".BatchVectorStruct.anyOf")
VECTOR_INPUT_ANY_OF=$(echo $SCHEMAS | jq -r ".VectorInput.anyOf")

MODIFIED_VECTOR_ANY_OF=$(jq -s 'add' <(echo "$VECTOR_ANY_OF") <(echo "$DOCUMENT_FLAT_EXTENSION"))
MODIFIED_VECTOR_STRUCT_ANY_OF=$(jq -s 'add' <(echo "$VECTOR_STRUCT_ANY_OF") <(echo "$DOCUMENT_FLAT_EXTENSION"))
MODIFIED_BATCH_VECTOR_STRUCT_ANY_OF=$(jq -s 'add' <(echo "$BATCH_VECTOR_STRUCT_ANY_OF") <(echo "$DOCUMENT_ARRAY_EXTENSION"))
MODIFIED_VECTOR_INPUT_ANY_OF=$(jq -s 'add' <(echo "$VECTOR_INPUT_ANY_OF") <(echo "$DOCUMENT_FLAT_EXTENSION"))

MODIFIED=$(cat <<EOF
{
  "components": {
    "schemas": {
      "VectorStruct": {"anyOf": $MODIFIED_VECTOR_STRUCT_ANY_OF},
      "BatchVectorStruct": {"anyOf": $MODIFIED_BATCH_VECTOR_STRUCT_ANY_OF},
      "VectorInput": {"anyOf": $MODIFIED_VECTOR_INPUT_ANY_OF},
      "Vector": {"anyOf": $MODIFIED_VECTOR_ANY_OF},
      "Document": $DOCUMENT_TYPE
    }
  }
}
EOF
)

validate_json "$merged_json"
if [ $? -ne 0 ]; then
  echo "Invalid JSON: $merged_json"
  exit 1
fi

merged_json=$(jq -s '.[0] * .[1]' <(echo "$OPENAPI_SOURCE") <(echo "$MODIFIED"))

validate_json "$merged_json"
if [ $? -ne 0 ]; then
  echo "Invalid JSON: $merged_json"
  exit 1
fi

echo $merged_json
