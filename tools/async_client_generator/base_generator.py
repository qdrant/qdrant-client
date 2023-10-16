import ast
from typing import List

from tools.async_client_generator.config import AUTOGEN_WARNING_MESSAGE


class BaseGenerator:
    def __init__(self) -> None:
        self.transformers: List[ast.NodeTransformer] = []

    def generate(self, code: str) -> str:
        nodes = ast.parse(code)

        for transformer in self.transformers:
            nodes = transformer.visit(nodes)

        return AUTOGEN_WARNING_MESSAGE + ast.unparse(nodes)
