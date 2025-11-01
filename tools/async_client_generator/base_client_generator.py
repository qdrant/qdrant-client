from typing import Optional

from tools.async_client_generator.base_generator import BaseGenerator
from tools.async_client_generator.transformers import (
    ClassDefTransformer,
    ConstantTransformer,
    FunctionDefTransformer,
    ImportFromTransformer,
    ImportTransformer,
    CallTransformer,
)


class BaseClientGenerator(BaseGenerator):
    def __init__(
        self,
        keep_sync: Optional[list[str]] = None,
        class_replace_map: Optional[dict[str, str]] = None,
        constant_replace_map: Optional[dict[str, str]] = None,
        import_replace_map: Optional[dict[str, str]] = None,
        rename_methods: Optional[dict[str, str]] = None,
    ):
        super().__init__()

        self.transformers.append(ImportTransformer(import_replace_map=import_replace_map))
        self.transformers.append(ImportFromTransformer(import_replace_map=import_replace_map))
        self.transformers.append(
            FunctionDefTransformer(
                keep_sync=keep_sync,
                rename_methods=rename_methods,
                class_replace_map=class_replace_map,
            )
        )
        self.transformers.append(ClassDefTransformer(class_replace_map=class_replace_map))
        self.transformers.append(ConstantTransformer(constant_replace_map=constant_replace_map))
        self.transformers.append(CallTransformer(async_methods=self.async_methods))

    @property
    def async_methods(self) -> list[str]:
        # as we do not have a class (yet) to extract async methods from, we return a hardcoded list
        return ["close"]


if __name__ == "__main__":
    from tools.async_client_generator.config import CLIENT_DIR, CODE_DIR

    with open(CLIENT_DIR / "client_base.py", "r") as source_file:
        code = source_file.read()

    # Parse the code into an AST
    base_client_generator = BaseClientGenerator(
        keep_sync=["__init__", "upload_records", "upload_collection", "upload_points", "migrate"],
        class_replace_map={
            "QdrantBase": "AsyncQdrantBase",
            "AbstractContextManager": "AbstractAsyncContextManager",
        },
        constant_replace_map={"QdrantBase": "AsyncQdrantBase"},
        import_replace_map={"AbstractContextManager": "AbstractAsyncContextManager"},
        rename_methods={"__enter__": "__aenter__", "__exit__": "__aexit__"},
    )
    modified_code = base_client_generator.generate(code)

    with open(CODE_DIR / "async_client_base.py", "w") as target_file:
        target_file.write(modified_code)
