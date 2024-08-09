from typing import Dict, List, Optional

from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    ClassDefTransformer,
    ConstantTransformer,
    FunctionDefTransformer,
)


class BaseClientGenerator(BaseGenerator):
    def __init__(
        self,
        keep_sync: Optional[List[str]] = None,
        class_replace_map: Optional[Dict[str, str]] = None,
        constant_replace_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        self.transformers.append(FunctionDefTransformer(keep_sync=keep_sync))
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))
        self.transformers.append(ConstantTransformer(constant_replace_map=constant_replace_map))


if __name__ == "__main__":
    from tools.async_client_generator.config import CLIENT_DIR, CODE_DIR

    with open(CLIENT_DIR / "client_base.py", "r") as source_file:
        code = source_file.read()

    # Parse the code into an AST
    base_client_generator = BaseClientGenerator(
        keep_sync=["__init__", "upload_records", "upload_collection", "upload_points", "migrate"],
        class_replace_map={"QdrantBase": "AsyncQdrantBase"},
        constant_replace_map={"QdrantBase": "AsyncQdrantBase"},
    )
    modified_code = base_client_generator.generate(code)

    with open(CODE_DIR / "async_client_base.py", "w") as target_file:
        target_file.write(modified_code)
