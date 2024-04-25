from tools.async_client_generator.transformers import FunctionDefTransformer


class FastembedFunctionDefTransformer(FunctionDefTransformer):
    def _keep_sync(self, name: str) -> bool:
        # dumb approach:
        # does not check for async methods used inside methods, instead replaces every method that is not in keep_sync
        return name in self.keep_sync
