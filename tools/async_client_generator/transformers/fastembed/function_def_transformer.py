from ...transformers import FunctionDefTransformer


class FastembedFunctionDefTransformer(FunctionDefTransformer):
    def _keep_sync(self, name: str) -> bool:
        return name in self.keep_sync or name.startswith("_")
