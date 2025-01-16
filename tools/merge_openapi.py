import json
from typing import Any, Dict


def merge(source: Dict[str, Any], extension: Dict[str, Any]) -> Dict[str, Any]:
    """Merges two dictionaries, concatenating lists and merging dictionaries recursively.

    If a key exists in both dictionaries:
      - If the value is a dictionary, merge recursively.
      - If the value is a list, concatenate the lists.
      - Otherwise, replace the value from source_json with the value from extension_json.

    Args:
        source (Dict[str, Any]): The source dictionary to be merged into.
        extension (Dict[str, Any]): The dictionary to merge from.

    Returns:
        Dict[str, Any]: The merged dictionary.

    Raises:
        TypeError: If the value types do not match for lists or dictionaries.

    Example:
        >>> origin = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        >>> addition = {"a": 10, "b": [4, 5], "c": {"e": 5}}
        >>> merged = merge(origin, addition)
        >>> assert merged == {"a": 10, "b": [2, 3, 4, 5], "c": {"d": 4, "e": 5}}
    """
    for key, value in extension.items():
        if key in source:
            source_value = source[key]
            if isinstance(source_value, dict):
                if not isinstance(value, dict):
                    raise TypeError("Value of '{}' is not a dictionary".format(key))

                source[key] = merge(source_value, value)
            elif isinstance(source_value, list):
                if not isinstance(value, list):
                    raise TypeError("Value of '{}' is not a list".format(key))

                source_value.extend(value)

            else:
                source[key] = value
        else:
            source[key] = value
    return source


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_json", type=str, help="source openAPI file")
    parser.add_argument("structures_json", type=str, help="structures openAPI file")
    parser.add_argument(
        "output_json",
        type=str,
        help="File with the extended openAPI definition",
        default=None,
        nargs="?",
    )

    args = parser.parse_args()
    openapi_source_path = args.source_json
    structures_json_path = args.structures_json
    output_json_path = args.output_json
    with open(openapi_source_path, "r") as f:
        openapi_source = json.load(f)

    with open(structures_json_path, "r") as f:
        structures = json.load(f)

    output = merge(openapi_source, structures)
    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(output, f)
    else:
        print(json.dumps(output, indent=4))
