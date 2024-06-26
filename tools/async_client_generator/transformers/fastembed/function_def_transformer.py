from tools.async_client_generator.transformers import FunctionDefTransformer


class FastembedFunctionDefTransformer(FunctionDefTransformer):
    def _keep_sync(self, name: str) -> bool:
        # dump approach:
        # does not check for async methods used inside methods, instead replaces every method that is not in keep_sync
        # and does not start with "_" will be transformed to async
        return name in self.keep_sync or name.startswith("_")
