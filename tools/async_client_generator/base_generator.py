# type: ignore
import ast
from typing import List


class BaseGenerator:
    def __init__(self):
        self.transformers: List[ast.NodeTransformer] = []

    def generate(self, code: str) -> str:
        nodes = ast.parse(code)

        for transformer in self.transformers:
            nodes = transformer.visit(nodes)

        return ast.unparse(nodes)
